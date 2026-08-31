#!/usr/bin/env python3
"""요청 ⑦ Q1 — L17 을 test-free 기준으로도 고를 수 있었는가.

배경: 계층 선택(L8/L17/L23)은 direction_I 스윕에서 **test 기반**으로 이뤄졌다
(I_dinov3_uniform.py: X_te = cache["test_L{layer}"], 그 위 StratifiedKFold nested CV,
best_cfg = mean_fusion 최댓값). 260531_performance_validation_protocol.md:30 이 우려한
바로 그 자리이고, test-free 전환 때 재검증되지 않았다.

여기서는 **논문이 실제로 쓰는 test-free 기준**을 세 계층에 적용한다.
  (a) val 안정성 — 검증 정상 점수 꼬리의 |z_robust| 최댓값 (작을수록 좋다)
  (b) val 분리도 — 검증 정상 점수의 변동계수 CV (작을수록 z-정규화가 안정적이다)
test 는 참고용으로만 함께 낸다 — 판정에 쓰지 않는다.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CACHE = R / "cache/dinov3_multilayer_vitl16"
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
LAYERS = ["L8", "L17", "L23"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/layer_choice_testfree_check.json"


def knn_max(bank, X):
    b = torch.from_numpy(bank).to(DEV)
    q = torch.from_numpy(X.reshape(-1, X.shape[2]).astype("f4")).to(DEV)
    mins = []
    for i in range(0, q.shape[0], 2048):
        mins.append(torch.cdist(q[i:i + 2048], b).min(1).values)
    out = torch.cat(mins).cpu().numpy().reshape(X.shape[0], X.shape[1]).max(1)
    del b, q; torch.cuda.empty_cache()
    return out


def robust_tail(v):
    """검증 정상 점수 꼬리의 |z_robust| 최댓값. 논문이 채택 판정에 쓴 지표."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return float(np.abs((v - med) / max(mad, 1e-12)).max())


def main():
    z = lambda t, v: (t - v.mean()) / (v.std() + 1e-12)
    fixed = {}
    for cat in CATS:
        for s in SEEDS:
            ed = (R / "reports/phase0/ead_repro_npz" if s == 42
                  else R / "reports/phase0/efficient_ad_official_small/npz")
            ze = np.load(ed / f"scores_{cat}_seed{s}.npz", allow_pickle=True)
            zev = np.load(R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
                          / f"val_good_{cat}_seed{s}.npz", allow_pickle=True)
            base = (R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
                    / f"psad_scores_{cat}_seed{s}_hc_tta_merge")
            zc = np.load(f"{base}_test.npz", allow_pickle=True)
            zcv = np.load(f"{base}_val.npz", allow_pickle=True)
            fixed[(cat, s)] = {
                "lt": np.array([str(x) for x in ze["label_type"]]),
                "ead": z(np.asarray(ze["score"], float), np.asarray(zev[zev.files[0]], float)),
                "comp": z(np.asarray(zc["scores"], float), np.asarray(zcv["scores"], float))}

    def ls(score, lt):
        g = lt == "good"
        au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                           np.r_[score[g], score[m]]))
        return 0.5 * (au(lt == "logical") + au(lt == "structural"))

    res = {}
    for L in LAYERS:
        tails, cvs, fus, alone = [], [], [], []
        per_cat = {}
        for cat in CATS:
            c = np.load(CACHE / f"{cat}.npz")
            ntr = len(list((LOCO / cat / "train/good").glob("*.png")))
            tr_all = c[f"train_{L}"]
            tr, va = tr_all[:ntr].astype("f4"), tr_all[ntr:].astype("f4")
            te = c[f"test_{L}"].astype("f4")
            ct = {"tail": [], "cv": [], "fusion": [], "alone": []}
            for s in SEEDS:
                fl = tr.reshape(-1, tr.shape[2])
                if fl.shape[0] > 50000:
                    fl = fl[np.random.default_rng(s).choice(fl.shape[0], 50000, replace=False)]
                v, t = knn_max(fl, va), knn_max(fl, te)
                ct["tail"].append(robust_tail(v))
                ct["cv"].append(float(v.std() / max(abs(v.mean()), 1e-12)))
                f = fixed[(cat, s)]
                ct["fusion"].append(ls(f["ead"] + z(t, v) + f["comp"], f["lt"]))
                ct["alone"].append(ls(t, f["lt"]))
            per_cat[cat] = {k: float(np.mean(x)) for k, x in ct.items()}
            tails.append(per_cat[cat]["tail"]); cvs.append(per_cat[cat]["cv"])
            fus.append(per_cat[cat]["fusion"]); alone.append(per_cat[cat]["alone"])
            del c
        res[L] = {"val_tail_mean": float(np.mean(tails)), "val_tail_max": float(np.max(tails)),
                  "val_cv_mean": float(np.mean(cvs)),
                  "test_fusion_LS": float(np.mean(fus)), "test_pc_alone_LS": float(np.mean(alone)),
                  "per_cat": per_cat}

    pick_tf = min(LAYERS, key=lambda L: res[L]["val_tail_mean"])
    pick_te = max(LAYERS, key=lambda L: res[L]["test_fusion_LS"])
    out = {"layers": res,
           "testfree_criterion": "val 정상 점수 꼬리 |z_robust| 최댓값의 5범주 평균 (작을수록 채택)",
           "pick_by_testfree": pick_tf, "pick_by_test_fusion": pick_te,
           "agree": pick_tf == pick_te,
           "origin_of_L17": ("direction_I/I_dinov3_uniform.py 스윕. X_te = cache['test_L{layer}'] "
                             "위 StratifiedKFold nested CV 로 best_cfg = mean_fusion 최댓값을 골랐다 "
                             "(best_cfg 'L17_cs100k_k1_max', best_loco 0.9683). **test 기반 선택**이다."),
           "note": "test 열은 참고용이며 판정에 쓰지 않았다."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)

    print(f"  {'계층':6s}{'val 꼬리 평균':>14s}{'val 꼬리 최대':>14s}{'val CV':>10s}"
          f"{'  |':>4s}{'test 융합':>10s}{'PC 단독':>9s}")
    for L in LAYERS:
        r = res[L]
        m = "  ←" if L == pick_tf else ""
        print(f"  {L:6s}{r['val_tail_mean']:>14.3f}{r['val_tail_max']:>14.3f}{r['val_cv_mean']:>10.4f}"
              f"{'  |':>4s}{r['test_fusion_LS']:>10.4f}{r['test_pc_alone_LS']:>9.4f}{m}")
    print(f"\n  test-free 기준 선택: {pick_tf}   |   test 융합 기준 선택: {pick_te}   "
          f"→ {'일치' if pick_tf == pick_te else '불일치'}")
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
