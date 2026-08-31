#!/usr/bin/env python3
"""EAD-S 5-seed 재현 최종 판정 — 추출된 npz 로 집계·검수.

판정 항목:
  1. 5-seed mean±std vs 논문 표 5 (0.8966 ± 0.0027)
  2. splicing_connectors σ — 원본 percat 보고치 0.0008 의 과소추정 여부 확정
  3. 전 범주 σ 대조 (재현 vs 원본 percat_table_fill.json)
  4. 시드 간 점수 상관 — 시드가 실제로 변동을 만드는지 (1.0 이면 시드 미적용)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
NPZ = R / "reports/phase0/efficient_ad_official_small/npz"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44, 45, 46]
PAPER_MEAN, PAPER_STD = 0.8966, 0.0027


def ls_auc(s, lab, lt):
    return 0.5 * sum(roc_auc_score(lab[(lt == "good") | (lt == k)], s[(lt == "good") | (lt == k)])
                     for k in ["logical", "structural"])


def main():
    per = {}          # per[cat][seed] = LS
    scores = {}
    for cat in CATS:
        per[cat] = {}
        for sd in SEEDS:
            p = NPZ / f"scores_{cat}_seed{sd}.npz"
            if not p.exists():
                print(f"  !! {cat}/seed{sd} npz 없음")
                continue
            z = np.load(p, allow_pickle=True)
            per[cat][sd] = ls_auc(z["score"].astype(float), z["label"].astype(int),
                                  z["label_type"].astype(str))
            scores[(cat, sd)] = z["score"].astype(float)

    print("=" * 76)
    print("EAD-S 5-seed {42-46} 재현 판정")
    print("=" * 76)

    # 1. 5-cat mean LS per seed → mean±std
    seed_means = {sd: float(np.mean([per[c][sd] for c in CATS]))
                  for sd in SEEDS if all(sd in per[c] for c in CATS)}
    v = np.array(list(seed_means.values()))
    print(f"\n[1] 시드별 5-cat mean LS: "
          + " ".join(f"{sd}={m:.4f}" for sd, m in seed_means.items()))
    print(f"    재현: {v.mean():.4f} ± {v.std(ddof=1):.4f}  |  논문 표5: {PAPER_MEAN} ± {PAPER_STD}")
    dev = abs(v.mean() - PAPER_MEAN) / PAPER_STD
    print(f"    평균 편차 {dev:.1f}σ(논문 std 기준) → "
          f"{'PASS' if dev <= 2 else 'CHECK'}")

    # 2-3. 범주별 σ 대조
    pcf = R / "reports/countgd/percat_table_fill.json"
    orig_std = {c: json.load(open(pcf))["per_cat"][c]["EAD"]["std"] for c in CATS} \
        if pcf.exists() else {}
    orig_mean = {c: json.load(open(pcf))["per_cat"][c]["EAD"]["mean"] for c in CATS} \
        if pcf.exists() else {}
    print(f"\n[2] 범주별: 재현 mean±std(5-seed) vs 원본 보고 mean±std")
    verdicts = {}
    for c in CATS:
        vals = np.array([per[c][sd] for sd in SEEDS if sd in per[c]])
        rs = vals.std(ddof=1)
        os_ = orig_std.get(c, float("nan"))
        ratio = rs / os_ if os_ and os_ > 0 else float("inf")
        verdicts[c] = ratio
        print(f"    {c:22s} 재현 {vals.mean():.4f}±{rs:.4f} | 원본 "
              f"{orig_mean.get(c, float('nan')):.4f}±{os_:.4f} | σ비 {ratio:5.1f}x"
              f"{'  << σ 과소추정 의심' if ratio > 3 else ''}")

    # 4. 시드 간 상관 (splicing 대표)
    print(f"\n[3] 시드 간 원점수 상관 (splicing, seed42 기준)")
    base = scores.get(("splicing_connectors", 42))
    for sd in SEEDS[1:]:
        s = scores.get(("splicing_connectors", sd))
        if base is not None and s is not None and len(base) == len(s):
            r = float(np.corrcoef(base, s)[0, 1])
            print(f"    seed42 vs seed{sd}: r={r:.4f}"
                  f"{'  !! 동일(시드 미적용?)' if r > 0.9999 else ''}")

    out = {"seed_means": seed_means,
           "aggregate": {"mean": float(v.mean()), "std": float(v.std(ddof=1))},
           "per_cat": {c: {"mean": float(np.mean([per[c][sd] for sd in SEEDS if sd in per[c]])),
                           "std": float(np.std([per[c][sd] for sd in SEEDS if sd in per[c]], ddof=1)),
                           "orig_std": orig_std.get(c)} for c in CATS},
           "paper_ref": {"mean": PAPER_MEAN, "std": PAPER_STD}}
    op = R / "reports/phase0/repro_audit/ead_5seed_verdict.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {op}")


if __name__ == "__main__":
    main()
