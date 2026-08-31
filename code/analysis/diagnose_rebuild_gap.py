#!/usr/bin/env python3
"""재구축 PSAD 분기가 원본에 못 미치는 폭과 원인을 특정한다.

발단: 논문 프로토콜(검증 정상 z-정규화 + 등가중 1,1,1 + 합)으로 재구축 hc 를 재면
L+S 0.9650 이 나온다. 논문의 test-free 수치는 0.9751 이다. 프로토콜을 바꿔서 격차를
메우려 하면 그 순간 test 로 규칙을 고르는 셈이 되므로, 격차의 **출처**를 먼저 가른다.

세 가지를 순서대로 확인한다.
  A. 융합 하네스가 맞는가 — 원본 보존 점수로 논문 수치가 재현되는가
  B. 재구축 PSAD 가 원본과 같은 신호인가 — **val 정상 표본만으로** 상관·꼬리 비교
  C. 어느 이미지가 무너지는가 — val 이상치의 UNet 분할 클래스 구성

배경 사실:
  - 원본 hcp 점수(d2_out, val_scores/)는 2026-06-12 자로 보존되어 있다.
  - 원본 hc 점수(d2_out_hc)는 2026-06-30 생성이라 2026-06-15 백업에 없다 -> 소실.
  - 즉 논문의 test-free 0.9751 은 **hcp** 로 계산된 값이고, 헤드라인 분기 구성(hc)과
    다르다. hc 의 test-free 수치를 얻으려면 재구축밖에 방법이 없다.

test 라벨은 A 단계(이미 발표된 수치의 재현 확인)에서만 쓰고, B/C 는 val 전용이다.
설계 선택은 어느 단계에서도 test 를 보고 하지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
VS = R / "reports/countgd/val_scores"
REB = R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
UNET = R / "external/PSAD_official/LOCO_MVTec_AD/unet_seed42"
EADD = R / "reports/phase0"
V4 = R / "algml_v6_5/aupr_bootstrap/scores"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def ead(cat, seed=42):
    t = np.load(EADD / "efficient_ad_official_small/npz" / f"scores_{cat}_seed{seed}.npz")
    v = np.load(EADD / "efficient_ad_official_small" / f"{cat}_seed{seed}_v1" /
                "val_good_scores_v2.npz")
    return t["score"].astype(float), v["scores"].astype(float)


def psad_orig(cat, tag, ltypes):
    z = np.load(VS / f"psad_{cat}_{tag}_testraw.npz", allow_pickle=True)
    d = {(str(t), str(f)): s for s, t, f in zip(z["score"].astype(float), z["ltype"], z["fname"])}
    order = [(t, fn) for t in ["good", "logical", "structural"]
             for fn in sorted([k[1] for k in d if k[0] == t])]
    arr = np.array([d[o] for o in order])
    assert (np.array([o[0] for o in order]) == ltypes).all(), f"{cat} 정렬 불일치"
    v = np.load(VS / f"psad_{cat}_{tag}_val.npz", allow_pickle=True)["score"].astype(float)
    return arr, v


def maxz(x):
    med = np.median(x)
    iqr = max(float(np.subtract(*np.percentile(x, [75, 25]))), 1e-12)
    return float(np.max(np.abs((x - med) / iqr)))


def step_a():
    print("=" * 78)
    print("A. 융합 하네스 검증 — 원본 보존 점수로 논문 프로토콜 재현")
    print("=" * 78)
    print("논문은 EAD 를 3시드 평균으로 썼다. 시드 0/1234 의 EAD val 은 백업 누락으로")
    print("없으므로 여기서는 EAD seed42 단독을 쓴다 — 잡음 평균이 빠져 약 0.4pp 낮다.\n")
    out = {}
    for tag in ["official", "seed0", "seed42", "seed1234"]:
        ls = []
        for cat in CATS:
            z = np.load(V4 / f"scores_{cat}_seed42.npz", allow_pickle=True)
            lab, lt = z["label"], z["label_type"]
            et, ev = ead(cat)
            pt = np.load(VS / f"pc_{cat}_test_scores.npz")["test_trainonly"].astype(float)
            pv = np.load(VS / f"pc_{cat}_val.npz", allow_pickle=True)["score"].astype(float)
            st, sv = psad_orig(cat, tag, lt)
            f = sum((t - v.mean()) / max(v.std(), 1e-9)
                    for t, v in [(et, ev), (pt, pv), (st, sv)])
            ls.append(0.5 * sum(roc_auc_score(lab[(lt == "good") | (lt == k)],
                                              f[(lt == "good") | (lt == k)])
                                for k in ["logical", "structural"]))
        out[tag] = (float(np.mean(ls)), ls)
        print(f"  [{tag:9s}] L+S {np.mean(ls):.4f}   " +
              " ".join(f"{c[:5]} {v:.3f}" for c, v in zip(CATS, ls)))
    m = np.mean([out[t][0] for t in ["seed0", "seed42", "seed1234"]])
    print(f"\n  3시드 평균 {m:.4f} (논문 0.9751, EAD 3시드 평균 사용 시)")
    return out


def step_b():
    print("\n" + "=" * 78)
    print("B. 재구축 PSAD 가 원본과 같은 신호인가 — val 정상 표본만 (test 미조회)")
    print("=" * 78)
    print(f"{'범주':22s} {'n':>3s} {'Spearman':>9s} {'Pearson':>8s} "
          f"{'원본 CV':>8s} {'재구축 CV':>9s} {'원본 최대z':>9s} {'재구축 최대z':>11s}")
    for cat in CATS:
        o = np.load(VS / f"psad_{cat}_seed42_val.npz", allow_pickle=True)["score"].astype(float)
        r = np.load(REB / f"psad_scores_{cat}_seed42_hcp_val.npz",
                    allow_pickle=True)["scores"].astype(float)
        n = min(len(o), len(r))
        print(f"{cat:22s} {n:3d} {spearmanr(o[:n], r[:n]).statistic:9.3f} "
              f"{pearsonr(o[:n], r[:n]).statistic:8.3f} {o.std()/o.mean():8.3f} "
              f"{r.std()/r.mean():9.3f} {maxz(o):9.2f} {maxz(r):11.2f}")
    print("\n같은 hcp 구성인데 상관이 0.05~0.77 이고 screw_bag 꼬리가 1.89 -> 14.11 이다.")
    print("재구축은 원본과 다른 신호다. 격차의 출처는 융합 규칙이 아니라 이 분기다.")


def step_c(cats=("screw_bag", "splicing_connectors")):
    print("\n" + "=" * 78)
    print("C. 어느 이미지가 무너지는가 — val 이상치의 UNet 분할 (test 미조회)")
    print("=" * 78)
    for cat in cats:
        z = np.load(REB / f"psad_scores_{cat}_seed42_hc_val.npz", allow_pickle=True)
        s = z["scores"].astype(float); paths = z["paths"]
        med = np.median(s)
        iqr = max(float(np.subtract(*np.percentile(s, [75, 25]))), 1e-12)
        zz = (s - med) / iqr
        top = np.argsort(-np.abs(zz))[:3]
        print(f"\n--- {cat}: 최대 |z| {np.abs(zz).max():.2f}, "
              f"상위 3장 제외 시 {np.sort(np.abs(zz))[-4]:.2f}")
        trm = sorted((UNET / cat / "train/good").glob("*.png"))[:40]
        ref = np.stack([np.bincount(np.array(Image.open(p)).ravel(), minlength=32)[:8]
                        for p in trm])
        mu, sd = ref.mean(0), ref.std(0)
        for i in top:
            stem = Path(str(paths[i])).stem
            p = UNET / cat / "validation/good" / f"{stem}.png"
            if not p.exists():
                continue
            c = np.bincount(np.array(Image.open(p)).ravel(), minlength=32)[:8]
            dev = np.abs(c - mu) / (sd + 1)
            print(f"    {stem}.png z={zz[i]:7.2f} | " +
                  " ".join(f"c{k}={c[k]}" for k in range(6)) +
                  f" | 최대 편차 {dev.max():.1f}sigma (cls{dev.argmax()}, "
                  f"train 평균 {mu[dev.argmax()]:.0f}+-{sd[dev.argmax()]:.0f})")
        print(f"    train 기준: " + " ".join(f"c{k}={mu[k]:.0f}+-{sd[k]:.0f}" for k in range(6)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in which:
        step_a()
    if "b" in which:
        step_b()
    if "c" in which:
        step_c()
