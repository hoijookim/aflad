#!/usr/bin/env python3
"""§4.5 — 배포 구성 비교와 보정 표본 수. 저장된 per-image 점수만 쓴다(재학습 없음).

두 가지를 잰다.

1) **배포 구성 비교** — 현장이 고를 수 있는 구성을 같은 선언 예산에서 붙인다.
   융합(세 분기·단일 임계값) / 두 시스템 OR(재구성+구성) / 단일 분기 셋.
   임계값은 **전부 검증 정상 분위수**로 잡는다. test 는 결과 확인에만 쓴다.

   품목별 최적 분기(오라클)는 넣지 않는다 — test 라벨을 봐야 고를 수 있고,
   검증셋에 이상이 없어 라벨 없이는 고를 방법이 아예 없다. 그 사실 자체가 결과다.

2) **보정 표본 수** — 검증 정상을 K 장만 써서 z-정규화 통계와 임계값을 잡으면
   순위 품질(AUROC)과 선언 예산의 이행이 각각 어떻게 되는가.

출력: reports/countgd/deployment_operating_point.json
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
B = REPO / "submission/results/per_image_scores"
OUT = REPO / "reports/countgd/deployment_operating_point.json"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = (42, 43, 44)
BRANCH = {"재구성": 0, "패치 메모리": 1, "구성": 2}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(s, kind="mergesort"); s2 = s[o]; r = np.empty(len(s)); i = 0
    while i < len(s2):
        j = i
        while j + 1 < len(s2) and s2[j + 1] == s2[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    p = y == 1
    return (r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (y == 0).sum())


def load(cat, seed):
    # README: 융합은 seed42 에 재학습본(repro)을 쓴다 — 정본 seed42 는 val 과 출처가 섞였다
    d = "ead_s_seed42_repro" if seed == 42 else "ead_s"
    e = np.load(B / d / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
    lb = np.load(B / "ead_s" / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
    ev = np.load(B / "ead_s_val" / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
    pc = np.load(B / "pc_branch" / f"pc_L17_{cat}_seed{seed}.npz", allow_pickle=True)
    cv = np.load(B / "psad_composition" / f"psad_scores_{cat}_seed{seed}_hc_tta_merge_val.npz", allow_pickle=True)
    ct = np.load(B / "psad_composition" / f"psad_scores_{cat}_seed{seed}_hc_tta_merge_test.npz", allow_pickle=True)
    return (np.vstack([ev["scores"], pc["val"], cv["scores"]]),
            np.vstack([e["score"], pc["test"], ct["scores"]]),
            np.asarray(lb["label"]), np.array([str(x) for x in lb["label_type"]]))


D = {(c, s): load(c, s) for c in CATS for s in SEEDS}


def zed(V, T):
    mu, sd = V.mean(1, keepdims=True), V.std(1, ddof=0, keepdims=True)
    return (T - mu) / sd, (V - mu) / sd


def or_threshold(Zv, target):
    """두 분기 OR 의 **검증** 오경보가 target 이 되는 공통 분위수."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        thr = np.quantile(Zv, 1 - m, axis=1, keepdims=True)
        if (Zv > thr).any(0).mean() > target:
            hi = m
        else:
            lo = m
    return np.quantile(Zv, 1 - (lo + hi) / 2, axis=1, keepdims=True)


def alarms(name, Z, Zv, target):
    if name == "fusion":
        return Z.sum(0) > np.quantile(Zv.sum(0), 1 - target)
    if name == "or_two":
        idx = [BRANCH["재구성"], BRANCH["구성"]]
        return (Z[idx] > or_threshold(Zv[idx], target)).any(0)
    b = BRANCH[name]
    return Z[b] > np.quantile(Zv[b], 1 - target)


def deployment_table(target):
    res = {}
    for name in ["fusion", "or_two", "재구성", "패치 메모리", "구성"]:
        rows = []
        for s in SEEDS:
            fp, lo, st = [], [], []
            for c in CATS:
                V, T, lab, typ = D[(c, s)]
                Z, Zv = zed(V, T)
                al = alarms(name, Z, Zv, target); g = lab == 0
                fp.append(al[g].mean())
                lo.append(al[(lab == 1) & (typ == "logical")].mean())
                st.append(al[(lab == 1) & (typ == "structural")].mean())
            rows.append([np.mean(fp), np.mean(lo), np.mean(st)])
        m, sd = np.mean(rows, 0), np.std(rows, 0, ddof=1)
        res[name] = {"fpr": [round(m[0], 4), round(sd[0], 4)],
                     "logical": [round(m[1], 4), round(sd[1], 4)],
                     "structural": [round(m[2], 4), round(sd[2], 4)],
                     "mean_recall": round(m[1:].mean(), 4)}
    return res


def calibration_sweep(target=0.05, draws=200, seed0=0):
    rng = np.random.default_rng(seed0); out = {}
    for K in [5, 10, 20, 30, 50, None]:
        A, F = [], []
        for _ in range(1 if K is None else draws):
            ls, fp = [], []
            for s in SEEDS:
                lo, st = [], []
                for c in CATS:
                    V, T, lab, typ = D[(c, s)]
                    idx = slice(None) if K is None else rng.choice(V.shape[1], K, replace=False)
                    Z, Zv = zed(V[:, idx], T)
                    f = Z.sum(0); g = lab == 0
                    for key, acc in (("logical", lo), ("structural", st)):
                        m = g | ((lab == 1) & (typ == key))
                        acc.append(auroc(lab[m], f[m]))
                    fp.append((f > np.quantile(Zv.sum(0), 1 - target))[g].mean())
                ls.append((np.mean(lo) + np.mean(st)) / 2)
            A.append(np.mean(ls)); F.append(np.mean(fp))
        out["all" if K is None else K] = {
            "LS_AUROC": [round(float(np.mean(A)), 4), round(float(np.std(A, ddof=1)), 4) if len(A) > 1 else 0.0],
            "actual_fpr": round(float(np.mean(F)), 4)}
    return out



def fpr_needed(target=0.90, grid=2001):
    """두 축 모두 target 이상을 잡는 데 **필요한 최소 오경보율**.

    표의 왼쪽 열들과 읽는 방향이 반대다 — 그쪽은 비용을 고정하고 품질을 비교하고,
    이쪽은 품질을 고정하고 비용을 비교한다. 오경보 하나가 사람이 다시 보는 작업
    하나이므로 이 값이 같은 품질을 사는 데 드는 재검사량이다.

    주의: 이 값은 **점수의 성질**(ROC 곡선의 한 점)이지 배포 절차가 아니다.
    90% 지점을 찾으려면 라벨이 필요하다 — AUROC 가 test 에서 계산되는 것과 같다.
    실제 임계값은 검증 정상으로 정한다(deployment_table 참조).
    """
    out = {}
    for name in ["fusion", "or_two", "재구성", "패치 메모리", "구성"]:
        vals, miss = [], 0
        for s in SEEDS:
            for c in CATS:
                V, T, lab, typ = D[(c, s)]
                Z, _ = zed(V, T)
                g = lab == 0
                L = (lab == 1) & (typ == "logical"); S = (lab == 1) & (typ == "structural")
                sc = ([Z.sum(0)] if name == "fusion"
                      else [Z[BRANCH["재구성"]], Z[BRANCH["구성"]]] if name == "or_two"
                      else [Z[BRANCH[name]]])
                hit = None
                for q in np.linspace(0, 1, grid):
                    if len(sc) == 1:
                        al = sc[0] > (np.quantile(sc[0][g], 1 - q) if q > 0 else sc[0].max())
                    else:
                        lo, hi = 0.0, 1.0
                        for _ in range(40):
                            m = (lo + hi) / 2
                            th = [np.quantile(x[g], 1 - m) for x in sc]
                            if np.any([x > t for x, t in zip(sc, th)], 0)[g].mean() > q: hi = m
                            else: lo = m
                        th = [np.quantile(x[g], 1 - (lo + hi) / 2) for x in sc]
                        al = np.any([x > t for x, t in zip(sc, th)], 0)
                    if al[L].mean() >= target and al[S].mean() >= target:
                        hit = al[g].mean(); break
                if hit is None: miss += 1
                else: vals.append(hit)
        out[name] = {"fpr_needed": round(float(np.mean(vals)), 4) if vals else None,
                     "unreachable_cells": miss, "cells": len(SEEDS) * len(CATS)}
    return out


def main():
    res = {
        "note": "저장된 per-image 점수만 사용. 임계값은 전부 검증 정상 분위수 — test 는 결과 확인 전용.",
        "val_normals_per_category": {c: int(D[(c, 42)][0].shape[1]) for c in CATS},
        "deployment": {f"declared_{int(t*100)}pct": deployment_table(t) for t in (0.10, 0.05, 0.02)},
        "fpr_needed_for_both_axes": {f"target_{int(t*100)}pct": fpr_needed(t)
                                     for t in (0.90, 0.95)},
        "calibration_sweep": {f"declared_{int(t*100)}pct": calibration_sweep(t)
                              for t in (0.05, 0.02)},
        "omitted": ("품목별 최적 분기(오라클)는 제외 — test 라벨이 있어야 고를 수 있고, "
                    "검증셋에 이상이 없어 라벨 없이는 선택 자체가 불가능하다."),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for tag, tb in res["deployment"].items():
        print(f"\n[{tag}]")
        for k, v in tb.items():
            print(f"  {k:<12} fpr {v['fpr'][0]:.3f}  logical {v['logical'][0]:.3f}  "
                  f"structural {v['structural'][0]:.3f}  mean {v['mean_recall']:.3f}")
    for tag, tb in res["fpr_needed_for_both_axes"].items():
        print("")
        print("[fpr needed " + tag + "]")
        for k, v in tb.items():
            print("  %-12s %s  (미달 %d/%d)" % (k, v["fpr_needed"], v["unreachable_cells"], v["cells"]))
    for tag, sw in res["calibration_sweep"].items():
        print("")
        print("[calibration " + tag + "]")
        for k, v in sw.items():
            print("  K=%-4s AUROC %.4f   actual_fpr %.3f"
                  % (str(k), v["LS_AUROC"][0], v["actual_fpr"]))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
