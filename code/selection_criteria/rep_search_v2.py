#!/usr/bin/env python3
"""구성 표현 탐색 — 기준 v2(4가지 논리이상 모사)로 재평가.

v1(제거만)에서는 현행 "클래스별 면적"(V0)이 최선으로 보였다. 그런데 v2 로 유형을
넓히자 구멍이 드러났다: breakfast_box 의 duplicate 가 0.5837 로 거의 우연이다.
면적만 보면 부품이 하나 늘어난 것과 기존 부품이 조금 커진 것을 구분할 수 없고,
breakfast_box 는 시리얼·견과처럼 면적이 원래 크게 흔들리는 품목이 있다.

후보 표현 (모두 마스크에서 바로 계산, 학습 없음):
  V0 면적            클래스별 픽셀 수                                (현행)
  V4 면적+개수       + 클래스별 연결성분 수                          (개수 신호)
  V5 면적+개수+평균  + 클래스별 성분 평균 크기                        (크기 신호)
  V6 면적+개수+위치  + 클래스별 무게중심 (x,y)                        (배치 신호)
  V7 전체            면적+개수+평균크기+위치

test 는 어느 단계에서도 쓰지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import criterion_v2 as C2  # noqa: E402

SEG = R / "external/PSAD_official/LOCO_MVTec_AD"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PROBES = [0, 1, 2, 3, 4]


def _per_class(m, K):
    """클래스별 (면적, 성분수, 성분평균크기, 무게중심y, 무게중심x)."""
    A, N, S, Y, X = [], [], [], [], []
    h, w = m.shape
    for k in range(1, K):
        sel = m == k
        a = sel.sum()
        A.append(float(a))
        if a == 0:
            N.append(0.0); S.append(0.0); Y.append(0.0); X.append(0.0)
            continue
        comp, n = ndimage.label(sel)
        N.append(float(n))
        S.append(float(a) / max(n, 1))
        ys, xs = np.nonzero(sel)
        Y.append(float(ys.mean()) / h); X.append(float(xs.mean()) / w)
    return map(np.array, (A, N, S, Y, X))


REPS = {
    "V0 면적(현행)":      lambda A, N, S, Y, X: A,
    "V4 면적+개수":       lambda A, N, S, Y, X: np.r_[A, N],
    "V5 +평균크기":       lambda A, N, S, Y, X: np.r_[A, N, S],
    "V6 +위치":           lambda A, N, S, Y, X: np.r_[A, N, Y, X],
    "V7 전체":            lambda A, N, S, Y, X: np.r_[A, N, S, Y, X],
}


def load(seg_dir, cat):
    return [np.array(Image.open(p)) for p in sorted((SEG / seg_dir / cat).glob("pred_*.png"))]


def eval_rep(masks, rep_fn, K, kind, probes=PROBES, n_probe=120):
    X = np.stack([rep_fn(*_per_class(m, K)) for m in masks])
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Z = (X - mu) / sd

    def nn(q, ex):
        d = np.sqrt(((Z - q) ** 2).sum(1)); d[ex] = np.inf
        return d.min()

    normal = np.array([nn(Z[i], i) for i in range(len(Z))])
    fn = C2.PERTURBS[kind]
    vals = []
    for ps in probes:
        rng = np.random.default_rng(ps)
        pert = []
        for i in rng.choice(len(masks), min(n_probe, len(masks)), replace=False):
            mm = fn(masks[i], rng)
            if mm is None:
                continue
            pert.append(nn((rep_fn(*_per_class(mm, K)) - mu) / sd, i))
        if len(pert) < 10:
            return None, None
        vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                  np.r_[normal, pert]))
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="csad_pseudo_seg")
    ap.add_argument("--cats", default=",".join(CATS))
    a = ap.parse_args()
    out = {}
    for cat in a.cats.split(","):
        masks = load(a.seg, cat)
        if not masks:
            continue
        K = max(m.max() for m in masks) + 1
        print(f"\n=== {cat} (K={K-1}, {len(masks)}장) ===", flush=True)
        print(f"{'표현':18s} " + " ".join(f"{k:>10s}" for k in C2.PERTURBS) + f"{'평균':>10s}")
        out[cat] = {}
        for nm, fn in REPS.items():
            row = {}
            for kind in C2.PERTURBS:
                m, s = eval_rep(masks, fn, K, kind)
                row[kind] = {"mean": m, "std": s}
            got = [row[k]["mean"] for k in C2.PERTURBS if row[k]["mean"] is not None]
            row["평균"] = float(np.mean(got)) if got else None
            out[cat][nm] = row
            print(f"{nm:18s} " + " ".join(
                ("{:>10s}".format("--") if row[k]["mean"] is None else f"{row[k]['mean']:10.4f}")
                for k in C2.PERTURBS) + f"{row['평균']:10.4f}", flush=True)
        best = max(out[cat], key=lambda n: out[cat][n]["평균"])
        print(f"  -> 최선: {best} ({out[cat][best]['평균']:.4f}, "
              f"현행 대비 {out[cat][best]['평균'] - out[cat]['V0 면적(현행)']['평균']:+.4f})")

    print("\n" + "=" * 74)
    print(f"{'표현':18s} {'5범주 평균':>11s}   범주별 현행 대비")
    for nm in REPS:
        ms = [out[c][nm]["평균"] for c in out]
        dd = " ".join(f"{c[:5]} {out[c][nm]['평균'] - out[c]['V0 면적(현행)']['평균']:+.3f}"
                      for c in out)
        print(f"{nm:18s} {np.mean(ms):11.4f}   {dd}")
    p = R / f"reports/countgd/rep_search_v2_{a.seg}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
