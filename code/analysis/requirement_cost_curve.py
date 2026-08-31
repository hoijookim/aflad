#!/usr/bin/env python3
"""§4.5 — 검출 요건별 비용 곡선. 저장된 per-image 점수만 쓴다(재학습 없음).

**왜 운영 지점을 고르지 않는가.** 초안은 "오경보율 5% 를 선언했을 때" 로 표를 세웠는데,
5% 에 근거가 없었다. 이 분야에 이미지 수준 운영 지점의 표준은 없다 — 낮은 거짓양성률
구간을 중시하는 관례는 있으나(예: MVTec LOCO 는 국소화 지표를 FPR 5% 까지만 적분한다)
그것은 **적분 상한**이지 운영 지점이 아니고, 픽셀 수준 국소화의 값이다.

그래서 점을 고르지 않는다. **검출 요건 t 를 훑으면서 각 배포 방식이 요구하는 최소
오경보율**을 재면 임의 상수가 사라진다. AUROC 가 곡선 전체를 요약하듯 이것도 곡선의
성질이다.

계산:
  단일 점수  두 축 모두 t 이상이 되는 최대 임계값은 min(Q_L(1-t), Q_S(1-t)) 이므로
             닫힌 형태로 구한다. 오경보율 = 그 임계값을 넘는 정상 이미지 비율.
  OR         두 분기에 같은 분위수를 적용하고 그 분위수를 이분 탐색한다.

집계는 논문 관례와 같다 — 카테고리별로 계산해 다섯 카테고리를 평균한 뒤 3시드 평균±std.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
B = REPO / "submission/results/per_image_scores"
OUT = REPO / "reports/countgd/requirement_cost_curve.json"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = (42, 43, 44)
IDX = {"재구성": 0, "패치 메모리": 1, "구성": 2}
MODES = ["융합", "두 시스템 OR", "재구성", "패치 메모리", "구성"]
TS = [0.80, 0.85, 0.90, 0.95]


def load(cat, seed):
    # README: 융합은 seed42 에 재학습본을 쓴다 — 정본 seed42 는 val 과 출처가 섞였다
    d = "ead_s_seed42_repro" if seed == 42 else "ead_s"
    e = np.load(B / d / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
    lb = np.load(B / "ead_s" / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
    ev = np.load(B / "ead_s_val" / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
    pc = np.load(B / "pc_branch" / f"pc_L17_{cat}_seed{seed}.npz", allow_pickle=True)
    cv = np.load(B / "psad_composition" / f"psad_scores_{cat}_seed{seed}_hc_tta_merge_val.npz", allow_pickle=True)
    ct = np.load(B / "psad_composition" / f"psad_scores_{cat}_seed{seed}_hc_tta_merge_test.npz", allow_pickle=True)
    V = np.vstack([ev["scores"], pc["val"], cv["scores"]])
    T = np.vstack([e["score"], pc["test"], ct["scores"]])
    mu, sd = V.mean(1, keepdims=True), V.std(1, ddof=0, keepdims=True)
    return (T - mu) / sd, np.asarray(lb["label"]), np.array([str(x) for x in lb["label_type"]])


Z = {(c, s): load(c, s) for c in CATS for s in SEEDS}


def required_fpr(mode, cat, seed, t, axes=("logical", "structural")):
    z, lab, typ = Z[(cat, seed)]
    g = lab == 0
    masks = [(lab == 1) & (typ == a) for a in axes]
    if mode == "두 시스템 OR":
        a, b = z[IDX["재구성"]], z[IDX["구성"]]
        lo, hi = 0.0, 1.0
        for _ in range(50):
            m = (lo + hi) / 2
            al = (a > np.quantile(a[g], 1 - m)) | (b > np.quantile(b[g], 1 - m))
            if all(al[k].mean() >= t for k in masks):
                hi = m
            else:
                lo = m
        al = (a > np.quantile(a[g], 1 - hi)) | (b > np.quantile(b[g], 1 - hi))
        return float(al[g].mean())
    sc = z.sum(0) if mode == "융합" else z[IDX[mode]]
    tau = min(np.quantile(sc[k], 1 - t) for k in masks)   # 모든 축이 t 이상이 되는 최대 임계값
    return float((sc[g] > tau).mean())


def table(ts=TS, axes=("logical", "structural")):
    out = {}
    for m in MODES:
        row = []
        for t in ts:
            per_seed = [np.mean([required_fpr(m, c, s, t, axes) for c in CATS]) for s in SEEDS]
            row.append([round(float(np.mean(per_seed)), 4), round(float(np.std(per_seed, ddof=1)), 4)])
        out[m] = row
    return out


def dominance(lo=0.70, hi=0.98, step=0.01):
    grid = np.round(np.arange(lo, hi + 1e-9, step), 2)
    win = 0
    for t in grid:
        v = {m: np.mean([required_fpr(m, c, s, t) for c in CATS for s in SEEDS]) for m in MODES}
        if all(v["융합"] <= v[m] + 1e-12 for m in MODES if m != "융합"):
            win += 1
    return {"range": [lo, hi], "step": step, "points": int(len(grid)), "fusion_lowest_at": int(win)}


def main():
    res = {
        "note": "두 축을 모두 t 이상 잡는 데 필요한 최소 오경보율. 운영 지점을 고르지 않는다.",
        "aggregation": "카테고리별 계산 → 5범주 평균 → 3시드 평균±std (논문 관례와 동일)",
        "both_axes": table(),
        "dominance": dominance(),
        "single_axis_at_90": {a: {m: round(float(np.mean(
            [required_fpr(m, c, s, 0.90, (a,)) for c in CATS for s in SEEDS])), 4) for m in MODES}
            for a in ("logical", "structural")},
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("%-13s " % "배포 방식" + " ".join("%-15s" % ("t=%.2f" % t) for t in TS))
    for m in MODES:
        print("%-13s " % m + " ".join("%.3f ± %.3f " % tuple(v) for v in res["both_axes"][m]))
    d = res["dominance"]
    print("\n요건 %.2f~%.2f (%d 지점) 중 융합 최소: %d" % (d["range"][0], d["range"][1], d["points"], d["fusion_lowest_at"]))
    for a, v in res["single_axis_at_90"].items():
        print("  [%s 한 축 90%%] " % a + "  ".join("%s %.3f" % (m, x) for m, x in v.items()))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
