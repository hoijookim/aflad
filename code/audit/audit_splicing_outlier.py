#!/usr/bin/env python3
"""적대적 심층조사 — splicing_connectors 재현 편차(10.5σ)의 정체.

가설 후보:
  H1 학습 자체가 잘못됨(버그)          → 점수 상관이 낮고 분포가 깨져 있을 것
  H2 환경 차이로 인한 정상적 재학습 변동 → 상관은 높고 소수 경계 이미지에서 순위만 뒤바뀜
  H3 원본 σ=0.0008 자체가 과소추정      → 다른 시드(43,44) 재현치의 산포로 검증
어느 쪽인지 데이터로 가른다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/workspace/ai-vision-research/scripts/phase0")
R = Path("/workspace/ai-vision-research")
NPZ = R / "reports/phase0/efficient_ad_official_small/npz"
CAT = "splicing_connectors"


def axis_auc(s, lab, lt, k):
    m = (lt == "good") | (lt == k)
    return roc_auc_score(lab[m], s[m])


def main():
    from ead_infer_seeds import infer  # 재현 점수 재계산(결정적)

    old = np.load(NPZ / f"scores_{CAT}_seed42.npz", allow_pickle=True)
    o_s, lab, lt = old["score"].astype(float), old["label"].astype(int), old["label_type"].astype(str)
    n_s, n_lab, n_lt = infer(CAT, 42)
    assert np.array_equal(lab, n_lab) and np.array_equal(lt, n_lt), "정렬 불일치"

    print("=" * 74)
    print(f"splicing_connectors 편차 심층조사 (n={len(o_s)})")
    print("=" * 74)

    print("\n[1] 축별 분해 — 편차가 어느 축에서 오는가")
    for k in ["logical", "structural"]:
        a_o, a_n = axis_auc(o_s, lab, lt, k), axis_auc(n_s, lab, lt, k)
        print(f"  {k:11s}: 정본 {a_o:.4f}  재현 {a_n:.4f}  Δ {a_n-a_o:+.4f}")
    ls_o = 0.5 * sum(axis_auc(o_s, lab, lt, k) for k in ["logical", "structural"])
    ls_n = 0.5 * sum(axis_auc(n_s, lab, lt, k) for k in ["logical", "structural"])
    print(f"  {'L+S':11s}: 정본 {ls_o:.4f}  재현 {ls_n:.4f}  Δ {ls_n-ls_o:+.4f}")

    print("\n[2] 점수 분포 건전성 (H1 반증: 분포가 깨졌는가)")
    for nm, s in [("정본", o_s), ("재현", n_s)]:
        print(f"  {nm}: min {s.min():.4f} max {s.max():.4f} mean {s.mean():.4f} "
              f"std {s.std():.4f} 고유값 {len(np.unique(s))}/{len(s)}")
    r = float(np.corrcoef(o_s, n_s)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(o_s, n_s).statistic)
    print(f"  Pearson {r:.4f} / Spearman {rho:.4f}")

    print("\n[3] 순위 교란 규모 (H2: 소수 경계 이미지만 뒤바뀌었는가)")
    ro, rn = np.argsort(np.argsort(o_s)), np.argsort(np.argsort(n_s))
    d = np.abs(ro - rn)
    print(f"  순위변동 중앙 {np.median(d):.0f} / 90퍼센타일 {np.percentile(d,90):.0f} / 최대 {d.max()}")
    print(f"  |순위변동| > 10% (={0.1*len(o_s):.0f}) 인 이미지: {(d > 0.1*len(o_s)).sum()}장")

    print("\n[4] good 대비 이상 점수 분리도 (정상 동작 확인)")
    for nm, s in [("정본", o_s), ("재현", n_s)]:
        g, a = s[lt == "good"], s[lt != "good"]
        print(f"  {nm}: good 평균 {g.mean():.4f} / 이상 평균 {a.mean():.4f} "
              f"/ 분리비 {a.mean()/max(g.mean(),1e-9):.3f}")

    print("\n[5] H3 검증 — 재현 시드 간 산포 (원본 σ=0.0008 이 과소추정인가)")
    ck = R / "reports/phase0/efficient_ad_official_small_seeds_std"
    vals = {}
    for sd in [42, 43, 44, 45, 46]:
        if (ck / f"{CAT}_seed{sd}/trainings/mvtec_loco/{CAT}/teacher_final.pth").exists():
            s2, l2, t2 = infer(CAT, sd)
            vals[sd] = 0.5 * sum(axis_auc(s2, l2, t2, k) for k in ["logical", "structural"])
            print(f"  재현 seed {sd}: L+S {vals[sd]:.4f}", flush=True)
    if len(vals) >= 2:
        v = np.array(list(vals.values()))
        print(f"  → 재현 시드 산포: mean {v.mean():.4f}, std(ddof=1) {v.std(ddof=1):.4f} "
              f"(원본 보고 std 0.0008)")
        print(f"  → 원본 σ 과소추정 여부: 재현 σ 가 {v.std(ddof=1)/0.0008:.1f}배")

    out = R / "_logs/splicing_outlier_audit.json"
    json.dump({"axis": {k: {"canonical": axis_auc(o_s, lab, lt, k),
                            "repro": axis_auc(n_s, lab, lt, k)} for k in ["logical", "structural"]},
               "pearson": r, "spearman": rho,
               "repro_seed_LS": {str(k): float(v) for k, v in vals.items()}},
              open(out, "w"), indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
