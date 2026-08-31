#!/usr/bin/env python3
"""요청 ⑪-1 — ComAD 의 **시드별 축 AUROC** 산출 (지금까지 "산출 불가" 였던 것).

## 왜 필요한가

표 2 에서 **ComAD 만 `±` 없이** 실립니다. 다른 여덟 행은 축별 표준편차가 채워졌는데
ComAD 만 비어 있습니다.

260817 회신이 그 이유를 이렇게 적었습니다:

> "시드별 L+S std 는 ComAD per-image 점수가 이 머신에 없어 산출 불가
>  (external/ROMAD_baselines 에 ComAD 디렉터리 없음)."

**그 진단은 그 머신에서는 맞았지만 여기서는 틀립니다.** 이 5090 에는
`external/ROMAD_baselines/ComAD/output_scores_seed{42,43,44}/` 가 있고,
각 1,568개(5범주 × test 전 이미지)의 `NNN_scores.json` 에 `img_level_score` 가
그대로 남아 있습니다.

## 무엇을 검증하나

산출만 하고 끝내면 "이 점수가 논문 값을 만든 그 점수인가" 가 안 걸립니다. 그래서
**먼저 재현부터 확인**합니다 — 이 per-image 점수로 계산한 3시드 평균이
`ls_canonical_scores.json`(= 논문 0.7987 의 출처)과 자릿수까지 맞는지 봅니다.

  1. per-cat pooled(전 이상 vs 정상 AUROC) 이 `comad_3seed.json` 의 per_seed 와 일치하는가
  2. per-cat logical/structural 3시드 평균이 `ls_canonical` 과 일치하는가
  3. 그러고 나서 비어 있던 **시드별 산포**를 낸다

1·2 가 맞아야 3 을 표에 넣을 수 있습니다. 안 맞으면 다른 실행의 점수라는 뜻입니다.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
SC = R / "external/ROMAD_baselines/ComAD"
CANON = R / "reports/path_y/metric_unify/ls_canonical_scores.json"
POOLED = R / "reports/path_y/comad/comad_3seed.json"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
SPLITS = {"good": "good", "logical_anomalies": "logical",
          "structural_anomalies": "structural"}
OUT = R / "reports/countgd/comad_axis_perseed.json"


def load_scores(seed, cat):
    """{라벨: [점수...]} — 파일명 순으로 읽는다."""
    out = {}
    for d, lab in SPLITS.items():
        p = SC / f"output_scores_seed{seed}" / cat / "test" / d
        vals = []
        for f in sorted(p.glob("*_scores.json")):
            j = json.loads(f.read_text())["img_level_score"]
            vals.append(float(j["__ndarray__"] if isinstance(j, dict) else j))
        out[lab] = np.asarray(vals, float)
    return out


def au(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))],
                               np.r_[neg, pos]))


def main():
    canon = json.loads(CANON.read_text())
    cc = next(v for k, v in _walk(canon) if k == "ComAD")
    pooled_ref = json.loads(POOLED.read_text())["per_seed_per_cat"]

    per = {}
    for cat in CATS:
        per[cat] = {}
        for s in SEEDS:
            v = load_scores(s, cat)
            per[cat][s] = {
                "n": {k: int(len(x)) for k, x in v.items()},
                "pooled": au(v["good"], np.r_[v["logical"], v["structural"]]),
                "logical": au(v["good"], v["logical"]),
                "structural": au(v["good"], v["structural"])}
            per[cat][s]["LS"] = 0.5 * (per[cat][s]["logical"] + per[cat][s]["structural"])

    # ── 검증 1: pooled 가 기존 3시드 기록과 맞는가 ─────────────────────
    d1 = []
    for cat in CATS:
        for s in SEEDS:
            ref = pooled_ref[f"seed{s}"][cat]
            d1.append(abs(per[cat][s]["pooled"] - ref))
    print(f"\n=== 검증 1 — pooled vs comad_3seed.json (15조합) ===")
    print(f"  최대 절대차 {max(d1):.3g}   {'PASS' if max(d1) < 1e-9 else 'FAIL'}")

    # ── 검증 2: 축별 3시드 평균이 정본과 맞는가 ────────────────────────
    d2 = []
    print(f"\n=== 검증 2 — 축별 3시드 평균 vs ls_canonical ===")
    print(f"{'범주':<22}{'logical':>22}{'structural':>22}")
    for cat in CATS:
        lo = np.mean([per[cat][s]["logical"] for s in SEEDS])
        st = np.mean([per[cat][s]["structural"] for s in SEEDS])
        rl, rs = cc["per_cat"][cat]["logical"], cc["per_cat"][cat]["structural"]
        d2 += [abs(lo - rl), abs(st - rs)]
        print(f"{cat:<22}{lo:>10.6f}/{rl:<11.6f}{st:>10.6f}/{rs:<11.6f}")
    print(f"  최대 절대차 {max(d2):.3g}   {'PASS' if max(d2) < 1e-9 else 'FAIL'}")
    ok = max(d1) < 1e-9 and max(d2) < 1e-9

    # ── 산출 3: 비어 있던 시드별 산포 ──────────────────────────────────
    agg = {}
    print(f"\n=== 산출 — ComAD 축별 시드 산포 (범주 평균 후 3시드, 논문 ± 정의) ===")
    for k in ("logical", "structural", "LS"):
        pc = [float(np.mean([per[c][s][k] for c in CATS])) for s in SEEDS]
        agg[k] = {"per_seed": pc, "mean": float(np.mean(pc)),
                  "std": float(np.std(pc, ddof=1))}
        print(f"  {k:<12}{agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}    "
              f"s42 {pc[0]:.4f}  s43 {pc[1]:.4f}  s44 {pc[2]:.4f}")

    print(f"\n  논문 인쇄값: L+S 0.7987 · logical 0.8691 · structural 0.7282")
    print(f"  본 산출값  : L+S {agg['LS']['mean']:.4f} · "
          f"logical {agg['logical']['mean']:.4f} · structural {agg['structural']['mean']:.4f}")

    OUT.write_text(json.dumps({
        "question": "ComAD 축별 시드 산포 — 260817 '산출 불가' 를 뒤집는다",
        "source": str(SC / "output_scores_seed{42,43,44}"),
        "verified_against_canonical": bool(ok),
        "max_abs_diff": {"pooled": float(max(d1)), "axis_mean": float(max(d2))},
        "aggregate": agg, "per_cat_per_seed": per}, ensure_ascii=False, indent=2))
    print(f"\n  [saved] {OUT}")


def _walk(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            if isinstance(v, dict):
                yield from _walk(v, pre + k + ".")


if __name__ == "__main__":
    main()
