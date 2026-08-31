#!/usr/bin/env python3
"""새 주 설정(test-free hc, 3-seed {42,43,44})용 보충 산출 4종.

논문 세션이 남긴 미완 목록에 대응한다:
  A. 표 3 융합 방법 비교      — 주 설정 vs raw-mean / best-single (RF 는 아래 주석 참조)
  B. §4.5 다중 지표           — AUPR, 최대 F1
  C. §4.6 쌍별 조합           — EAD+PC / EAD+comp / PC+comp vs 3소스
  D. 표 5 분기별 단독 열      — EAD / PC / PSAD-hc 단독, 범주별

전부 **이미 확정된 설정**(E5 혼동쌍 병합 + E7 회전 TTA screw_bag 한정, 논문 선언 프로토콜:
검증 정상 z-정규화 + 등가중 합)에서 산출하는 **보고용 수치**다. 설정을 고르는 데 test 를
쓰지 않는다 — 구성은 train/val 근거로 이미 확정됐다(PREREG 참조).

**RandomForest 행에 대하여**: 학습형 융합은 라벨된 이상 표본을 요구한다. 중첩 교차검증이
폐지되어 논문에 test 를 보는 절차가 0 이 된 상태이므로, RF 를 test-free 프로토콜로 산출할
방법이 없다. 이 스크립트는 RF 를 계산하지 않는다 — 표 3 에서 그 행을 빼거나 "라벨된 이상
표본이 있을 때의 참고치(프로토콜 다름)"로 명시해야 한다. 없는 수치를 만들지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import prereg_testfree as PT  # noqa: E402
import fuse_testfree_v2 as F  # noqa: E402

CATS = F.CATS
SEEDS = (42, 43, 44)
PT.PSAD_TAG = "_tta_merge"          # 확정 설정
PT.PSAD_KIND = "hc"
BRANCHES = ("ead", "pcL17", "psad_hc")
LABEL = {"ead": "EAD-S(재구성)", "pcL17": "DINOv3-L PC", "psad_hc": "구성(PSAD-hc)"}


def z(t, v):
    return (t - v.mean()) / max(v.std(), 1e-9)


def get(cat, seed):
    """분기별 (test, val) + 라벨."""
    _, _, lab, lt = PT.ead_one(cat, seed)
    d = {b: PT.branch(b, cat, seed) for b in BRANCHES}
    return d, lab, lt


def ls(score, lab, lt):
    return 0.5 * sum(roc_auc_score(lab[(lt == "good") | (lt == k)],
                                   score[(lt == "good") | (lt == k)])
                     for k in ("logical", "structural"))


def axes(score, lab, lt):
    return {k: float(roc_auc_score(lab[(lt == "good") | (lt == k)],
                                   score[(lt == "good") | (lt == k)]))
            for k in ("logical", "structural")}


def maxf1(score, lab):
    p, r, _ = precision_recall_curve(lab, score)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(f.max())


def agg(per_seed_vals):
    a = np.asarray(per_seed_vals, float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else None}


def main():
    out = {"config": {"psad": "hc + 혼동쌍 병합 + 회전 TTA(screw_bag)",
                      "fusion": "검증 정상 z-정규화 + 등가중 (1,1,1) 합",
                      "seeds": list(SEEDS)}}
    # 캐시: (cat, seed) -> (분기 dict, lab, lt)
    C = {(c, s): get(c, s) for c in CATS for s in SEEDS}

    # ---------- D. 분기별 단독 + 주 설정 (범주별) ----------
    print("=" * 78)
    print("D. 표 5 분기별 단독 열 + 주 설정 (범주별, 3-seed)")
    print("=" * 78)
    single = {}
    for b in BRANCHES + ("fusion3",):
        percat, tot = {}, []
        for c in CATS:
            vals = []
            for s in SEEDS:
                d, lab, lt = C[(c, s)]
                if b == "fusion3":
                    sc = sum(z(*d[x]) for x in BRANCHES)
                else:
                    sc = d[b][0]                     # 단독은 단조변환 불변 -> 원점수
                vals.append(ls(sc, lab, lt))
            percat[c] = agg(vals)
            tot.append(vals)
        allv = np.array(tot).mean(0)                 # 시드별 5범주 평균
        single[b] = {"per_cat": percat, **agg(allv)}
        nm = "3소스 융합(주 설정)" if b == "fusion3" else LABEL[b]
        sd = f" ± {single[b]['std']:.4f}" if single[b]["std"] is not None else ""
        print(f"{nm:22s} {single[b]['mean']:.4f}{sd}   " +
              " ".join(f"{c[:5]} {percat[c]['mean']:.3f}" for c in CATS))
    out["D_branch_standalone"] = single

    # ---------- C. 쌍별 조합 ----------
    print("\n" + "=" * 78)
    print("C. §4.6 쌍별 조합 (같은 프로토콜, 등가중)")
    print("=" * 78)
    pairs = {"EAD+PC": ("ead", "pcL17"), "EAD+구성": ("ead", "psad_hc"),
             "PC+구성": ("pcL17", "psad_hc")}
    pair_out = {}
    for nm, bs in pairs.items():
        vals = []
        for s in SEEDS:
            v = [ls(sum(z(*C[(c, s)][0][b]) for b in bs), *C[(c, s)][1:]) for c in CATS]
            vals.append(float(np.mean(v)))
        pair_out[nm] = agg(vals)
        print(f"{nm:12s} {pair_out[nm]['mean']:.4f} ± {pair_out[nm]['std']:.4f}   "
              f"3소스 대비 {pair_out[nm]['mean'] - single['fusion3']['mean']:+.4f}")
    pair_out["3소스"] = {k: single["fusion3"][k] for k in ("mean", "std")}
    out["C_pairwise"] = pair_out

    # ---------- A. 표 3 융합 방법 비교 ----------
    print("\n" + "=" * 78)
    print("A. 표 3 융합 방법 비교 (주 설정 기준)")
    print("=" * 78)
    methods = {}
    # 주 설정
    methods["주 설정 (val z-정규화 + 등가중 합)"] = {k: single["fusion3"][k]
                                              for k in ("mean", "std")}
    # raw-mean: 정규화 없이 원점수 평균 (스케일 차가 커 PC 가 지배한다)
    vals = []
    for s in SEEDS:
        v = [ls(np.mean([C[(c, s)][0][b][0] for b in BRANCHES], axis=0), *C[(c, s)][1:])
             for c in CATS]
        vals.append(float(np.mean(v)))
    methods["raw-mean (정규화 없음)"] = agg(vals)
    # best-single: 분기 단독 중 최고 (분기 선택은 test 를 보지 않는다 — 세 값을 모두 보고한다)
    best = max(BRANCHES, key=lambda b: single[b]["mean"])
    methods[f"최고 단일 분기 ({LABEL[best]})"] = {k: single[best][k] for k in ("mean", "std")}
    base = methods["주 설정 (val z-정규화 + 등가중 합)"]["mean"]
    for nm, v in methods.items():
        sd = f" ± {v['std']:.4f}" if v["std"] is not None else ""
        d = "" if nm.startswith("주 설정") else f"   Δ vs 주 설정 {v['mean']-base:+.4f}"
        print(f"{nm:34s} {v['mean']:.4f}{sd}{d}")
    print("\n※ RandomForest 등 학습형 융합은 라벨된 이상 표본을 요구한다. 중첩 교차검증이")
    print("  폐지되어 test-free 프로토콜로 산출할 방법이 없으므로 계산하지 않는다.")
    out["A_fusion_methods"] = methods

    # ---------- B. 다중 지표 ----------
    print("\n" + "=" * 78)
    print("B. §4.5 다중 지표 (주 설정)")
    print("=" * 78)
    met = {"AUROC(L+S)": [], "AUPR": [], "maxF1": [], "logical": [], "structural": []}
    for s in SEEDS:
        au, ap, f1, lgv, stv = [], [], [], [], []
        for c in CATS:
            d, lab, lt = C[(c, s)]
            sc = sum(z(*d[b]) for b in BRANCHES)
            au.append(ls(sc, lab, lt))
            ap.append(average_precision_score(lab, sc))
            f1.append(maxf1(sc, lab))
            ax = axes(sc, lab, lt); lgv.append(ax["logical"]); stv.append(ax["structural"])
        for k, v in (("AUROC(L+S)", au), ("AUPR", ap), ("maxF1", f1),
                     ("logical", lgv), ("structural", stv)):
            met[k].append(float(np.mean(v)))
    out["B_metrics"] = {k: agg(v) for k, v in met.items()}
    for k, v in out["B_metrics"].items():
        print(f"{k:14s} {v['mean']:.4f} ± {v['std']:.4f}")

    p = R / "reports/countgd/testfree_supplements_3seed.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
