#!/usr/bin/env python3
"""기준 v2 실행기 — 4가지 논리이상 모사에 대한 부품 구성 표현의 변별력.

먼저 **기준 자체를 검증**한다: pushpins 3x5(개선) vs 5x5(구) 를 v2 가 올바로 가르는가.
v1 은 0.9937 vs 0.7575 로 갈랐다. v2 가 그 판정을 유지하지 못하면 기준을 못 믿는다.

그 다음 각 범주에서 유형별 AUROC 를 낸다. 유형별로 갈리는 지점이 보이면 그것이
구성 분기의 실제 약점이다.

test 는 어느 단계에서도 쓰지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import criterion_v2 as C2  # noqa: E402

SEG = R / "external/PSAD_official/LOCO_MVTec_AD"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PROBES = [0, 1, 2, 3, 4]


def rep_classwise(m, K):
    return np.array([(m == k).sum() for k in range(1, K)], float)


def load(seg_dir, cat):
    return [np.array(Image.open(p)) for p in sorted((SEG / seg_dir / cat).glob("pred_*.png"))]


def eval_type(masks, kind, probes=PROBES, n_probe=120):
    K = max(m.max() for m in masks) + 1
    X = np.stack([rep_classwise(m, K) for m in masks])
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Z = (X - mu) / sd

    def nn(q, ex):
        d = np.sqrt(((Z - q) ** 2).sum(1)); d[ex] = np.inf
        return d.min()

    normal = np.array([nn(Z[i], i) for i in range(len(Z))])
    fn = C2.PERTURBS[kind]
    vals, rates = [], []
    for ps in probes:
        rng = np.random.default_rng(ps)
        pert, tried = [], 0
        for i in rng.choice(len(masks), min(n_probe, len(masks)), replace=False):
            tried += 1
            mm = fn(masks[i], rng)
            if mm is None:
                continue
            pert.append(nn((rep_classwise(mm, K) - mu) / sd, i))
        if len(pert) < 10:
            return None, None, 0.0
        rates.append(len(pert) / tried)
        vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                  np.r_[normal, pert]))
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1)), float(np.mean(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="csad_pseudo_seg")
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()

    # ---- 기준 자체 검증: pushpins 3x5 vs 5x5 ----
    if not a.skip_validate:
        alt = SEG / "csad_pseudo_seg_pins5x5"
        if alt.exists() and any((alt / "pushpins").glob("pred_*.png")):
            print("기준 검증 — pushpins 3x5(현행) vs 5x5(구)")
            for tag, sd in (("3x5", a.seg), ("5x5", "csad_pseudo_seg_pins5x5")):
                ms = load(sd, "pushpins")
                r = {k: eval_type(ms, k)[0] for k in C2.PERTURBS}
                print(f"  {tag}: " + " ".join(
                    f"{k} {('--' if v is None else f'{v:.4f}')}" for k, v in r.items()))
            print()
        else:
            print("기준 검증 생략 — 5x5 비교본(csad_pseudo_seg_pins5x5) 없음\n")

    out = {}
    print(f"{'범주':22s} " + " ".join(f"{k:>16s}" for k in C2.PERTURBS))
    for cat in a.cats.split(","):
        ms = load(a.seg, cat)
        if not ms:
            print(f"{cat:22s} (없음)")
            continue
        row = {}
        for k in C2.PERTURBS:
            m, s, rate = eval_type(ms, k)
            row[k] = {"mean": m, "std": s, "적용률": rate}
        out[cat] = row
        print(f"{cat:22s} " + " ".join(
            ("{:>16s}".format("--") if row[k]["mean"] is None
             else f"{row[k]['mean']:9.4f}±{row[k]['std']:.4f}") for k in C2.PERTURBS))

    vals = {k: [out[c][k]["mean"] for c in out if out[c][k]["mean"] is not None]
            for k in C2.PERTURBS}
    print("\n유형별 5범주 평균: " + " ".join(
        f"{k} {np.mean(v):.4f}" if v else f"{k} --" for k, v in vals.items()))
    p = R / f"reports/countgd/criterion_v2_{a.seg}.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
