#!/usr/bin/env python3
"""PSAD 재구축 차이의 **원인 분해** — 기계 탓인가 시드 탓인가, 아니면 둘 다 아닌가.

## 왜 필요한가 — 3자 대조가 원인을 재지 않았다

`psad_3way_compare.py` 는 260816(4090) vs 5090 재구축본의 분기 점수가
**Spearman 0.938** 로 갈린다는 것을 냈고, 원인을 "시드는 고정했으나 GPU 커널·cuDNN
알고리즘 선택이 기계마다 다르다" 로 적었다. **그런데 그것을 재지 않았다.**

같은 기계·같은 시드로 두 번 돌려도 0.938 이 나온다면 원인은 기계가 아니라
**학습 자체의 비결정성**이고, 그러면 "다른 기계에서 재현되지 않는다" 는 서술이 틀린다.

## 통제군은 이미 있다 — 추가 학습 0

두 재구축본 모두 **시드 42/43/44** 를 가지고 있다. 그러면 세 종류의 거리를 잴 수 있다:

| 이름 | 쌍 | 무엇이 다른가 |
|---|---|---|
| **cross_seed** | 같은 재구축본, 다른 시드 | 시드만 |
| **cross_rebuild** | 다른 재구축본, 같은 시드 | 기계 + 학습 비결정성 |
| (참고) cross_both | 다른 재구축본, 다른 시드 | 전부 |

## 판정 — 세 가지 결말이 구분된다

```
cross_rebuild ~ 1.0,  cross_seed < 1.0     -> 시드가 작동하고 기계는 무관.
                                              "재현된다" 가 맞다.
cross_seed < cross_rebuild < 1.0           -> 시드는 작동하나 기계가 추가 변동을 만든다.
                                              3자 대조의 서술이 맞다.
cross_rebuild ~ cross_seed                 -> **시드 고정이 듣지 않는다.**
                                              기계와 무관하게 매 학습이 다른 모델을 낸다.
                                              "기계 간 재현" 이 아니라 "학습 비결정성" 문제다.
```

세 번째면 3자 대조의 원인 서술을 정정해야 하고, 논문의 재현성 주장 형태도 달라진다 —
"다른 기계에서" 가 아니라 **"다시 학습하면"** 이 정확한 조건이 된다.

## 지표

순위 통계 두 가지를 병기한다. AUROC 가 순위 통계라 Spearman 이 본령이지만,
Kendall τ 는 이상치에 덜 끌려서 꼬리가 두꺼운 원점수에 보완이 된다.
분기 L+S 자체의 산포도 같은 분해로 낸다(점수가 달라도 지표는 같을 수 있다).
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
SRC = {"pkg_260816": R / "submission/results/per_image_scores/psad_composition",
       "rebuild_5090": R / "reports/countgd/psad_rebuild_5090"}
# 3-f 복제 실행(같은 기계·같은 시드, 실행 시점만 다름). 있으면 cross_replicate 를 추가한다.
REP2 = R / "reports/countgd/psad_rebuild_5090_rep2"
EAD_REPRO = R / "reports/phase0/ead_repro_npz"
EAD_STD = R / "reports/phase0/efficient_ad_official_small/npz"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
TAG = "hc_tta_merge"
OUT = R / "reports/countgd/psad_variance_decomp.json"


def load(src, cat, seed):
    b = f"{SRC[src]}/psad_scores_{cat}_seed{seed}_{TAG}"
    return np.load(f"{b}_test.npz", allow_pickle=True)["scores"].astype(float)


def labels(cat, seed):
    f = (EAD_REPRO if seed == 42 else EAD_STD) / f"scores_{cat}_seed{seed}.npz"
    return np.array([str(x) for x in np.load(f, allow_pickle=True)["label_type"]])


def ls(sc, lt):
    g = lt == "good"
    return 0.5 * sum(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones((lt == k).sum())],
                                   np.r_[sc[g], sc[lt == k]]) for k in ("logical", "structural"))


def main():
    S = {}          # (src, cat, seed) -> 점수
    M = {}          # (src, cat, seed) -> 분기 L+S
    for src in SRC:
        for cat in CATS:
            lt = labels(cat, SEEDS[0])
            for s in SEEDS:
                try:
                    v = load(src, cat, s)
                except FileNotFoundError:
                    continue
                S[(src, cat, s)] = v
                M[(src, cat, s)] = ls(v, lt)

    GROUPS = ["cross_seed", "cross_rebuild", "cross_both", "cross_replicate"]
    pairs = {g: [] for g in GROUPS}
    dls = {g: [] for g in GROUPS}

    # cross_replicate — 같은 기계·같은 시드(42)의 두 실행. 3-f 산출물이 있을 때만.
    if REP2.is_dir():
        for cat in CATS:
            f = REP2 / f"psad_scores_{cat}_seed42_{TAG}_test.npz"
            if not f.exists():
                continue
            b = np.load(f, allow_pickle=True)["scores"].astype(float)
            a = S.get(("rebuild_5090", cat, 42))
            if a is None:
                continue
            pairs["cross_replicate"].append(
                {"cat": cat, "a": "rebuild_5090/s42", "b": "rep2/s42",
                 "spearman": float(spearmanr(a, b).statistic),
                 "kendall": float(kendalltau(a, b).statistic)})
            dls["cross_replicate"].append(abs(M[("rebuild_5090", cat, 42)] - ls(b, labels(cat, 42))))
    for cat in CATS:
        for (s1, c1, k1), (s2, c2, k2) in itertools.combinations(
                [k for k in S if k[1] == cat], 2):
            same_src, same_seed = s1 == s2, k1 == k2
            grp = ("cross_seed" if same_src and not same_seed else
                   "cross_rebuild" if not same_src and same_seed else
                   "cross_both" if not same_src else None)
            if grp is None:
                continue
            a, b = S[(s1, c1, k1)], S[(s2, c2, k2)]
            pairs[grp].append({"cat": cat, "a": f"{s1}/s{k1}", "b": f"{s2}/s{k2}",
                               "spearman": float(spearmanr(a, b).statistic),
                               "kendall": float(kendalltau(a, b).statistic)})
            dls[grp].append(abs(M[(s1, c1, k1)] - M[(s2, c2, k2)]))

    print("\n=== PSAD 분기 점수 — 무엇이 다르면 얼마나 달라지나 ===\n")
    print(f"{'비교 종류':<16}{'쌍':>5}{'Spearman':>12}{'Kendall':>11}{'|Δ 분기 L+S|':>14}")
    agg = {}
    for grp, lab in [("cross_seed", "시드만 다름"), ("cross_rebuild", "재구축만 다름"),
                     ("cross_both", "둘 다 다름"), ("cross_replicate", "실행 시점만 다름")]:
        if not pairs[grp]:
            continue
        sp = np.mean([p["spearman"] for p in pairs[grp]])
        kt = np.mean([p["kendall"] for p in pairs[grp]])
        dl = np.mean(dls[grp])
        agg[grp] = {"n": len(pairs[grp]), "spearman": float(sp), "kendall": float(kt),
                    "abs_d_ls": float(dl)}
        print(f"{lab:<16}{len(pairs[grp]):>5}{sp:>12.4f}{kt:>11.4f}{dl:>14.4f}")

    # 3-f 판정 — 사전등록 docs/PREREG_psad_replicate_260823.md §3 의 규칙 그대로.
    cs0, rp = agg.get("cross_seed"), agg.get("cross_replicate")
    if cs0 and rp:
        Rv, Sv = rp["spearman"], cs0["spearman"]
        if Rv >= 0.99:
            verdict3f = ("(B) 결정적 — 기계 안에서는 시드가 실행을 고정한다. "
                         "기계 효과가 실재하며 3자 대조의 원래 서술이 결과적으로 맞다")
        elif abs(Rv - Sv) <= 0.02:
            verdict3f = ("(A) 비결정적 — 학습이 애초에 비결정적이다. 시드 고정이 재현을 "
                         "보장하지 않는다. 재현성 주장의 조건은 '다른 기계에서' 가 아니라 "
                         "'다시 학습하면' 이다")
        else:
            verdict3f = ("(C) 혼합 — 시드가 부분 작동하고 실행·기계가 추가 변동을 만든다. "
                         "양쪽을 보고하고 어느 쪽도 단정하지 않는다")
        print(f"\n=== 3-f 판정 (사전등록 §3) ===")
        print(f"  R(복제) = {Rv:.4f}   S(시드) = {Sv:.4f}   |R−S| = {abs(Rv-Sv):.4f}")
        print(f"  ==> {verdict3f}")
        print(f"  등록 예측 = (A) 비결정적 (R 0.93~0.95) -> "
              f"{'적중' if abs(Rv - Sv) <= 0.02 else '빗나감'}")
    else:
        verdict3f = None

    cs, cr = agg.get("cross_seed"), agg.get("cross_rebuild")
    if cs and cr:
        gap = cr["spearman"] - cs["spearman"]
        print(f"\n=== 판정 ===")
        print(f"  cross_rebuild − cross_seed (Spearman) = {gap:+.4f}")
        if cr["spearman"] > 0.99:
            v = "시드가 작동하고 기계는 무관 — '기계 간에 재현된다'"
        elif abs(gap) < 0.02:
            v = ("**시드 고정이 듣지 않는다** — 재구축 차이가 시드 차이와 같은 크기다. "
                 "기계 문제가 아니라 학습 비결정성이며, 정확한 조건은 '다시 학습하면' 이다")
        elif gap > 0:
            v = "시드는 작동하고 기계가 추가 변동을 만든다 — 3자 대조 서술이 맞다"
        else:
            v = "재구축 차이가 시드 차이보다 **크다** — 기계·환경 요인이 시드보다 지배적"
        print(f"  ==> {v}")

    print(f"\n=== 범주별 (Spearman) ===")
    print(f"{'범주':<22}{'시드만':>10}{'재구축만':>11}{'차':>10}")
    percat = {}
    for cat in CATS:
        a = [p["spearman"] for p in pairs["cross_seed"] if p["cat"] == cat]
        b = [p["spearman"] for p in pairs["cross_rebuild"] if p["cat"] == cat]
        if not a or not b:
            continue
        percat[cat] = {"cross_seed": float(np.mean(a)), "cross_rebuild": float(np.mean(b))}
        print(f"{cat:<22}{np.mean(a):>10.4f}{np.mean(b):>11.4f}{np.mean(b)-np.mean(a):>+10.4f}")

    OUT.write_text(json.dumps({
        "question": "PSAD 재구축 차이의 원인이 기계인가 시드인가 학습 비결정성인가",
        "verdict": v if cs and cr else "판정 불가(쌍 부족)",
        "verdict_3f": verdict3f,
        "prereg": "docs/PREREG_psad_replicate_260823.md",
        "mean": agg, "per_cat": percat, "pairs": pairs}, ensure_ascii=False, indent=2))
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
