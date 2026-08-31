#!/usr/bin/env python3
"""요청 ⑦ Q2 — greedy coreset 대신 무작위 표본을 쓴 근거.

논문은 "무작위 표본은 구현이 단순하면서 본 설정에서 충분한 성능을 보였다" 고 쓴다.
그런데 **현 설정(DINOv3-L L17)에서 둘을 비교한 실험이 없다.** 유일한 근거였던
random +2.13pp 는 DINOv2-B 시절이라 인용할 수 없다. 그래서 지금 만든다.

greedy k-center (PatchCore 의 coreset 선택법): 무작위 시작점 하나에서 출발해
"현재 뱅크로부터 가장 먼 점" 을 50,000 번 고른다. 무작위 표본과 뱅크 크기·k·집계를
모두 같게 두고 선택법만 바꾼다.

이건 **선택이 아니라 사후 절제**다 — 이미 무작위로 확정된 설정에 대한 근거 제시이며,
결과를 보고 설정을 바꾸지 않는다.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CACHE = R / "cache/dinov3_multilayer_vitl16"
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
LAYER = "L17"
BANK = 50000
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/coreset_greedy_vs_random.json"


def greedy_kcenter(X, n, seed):
    """PatchCore 의 greedy k-center. X: (N,d) torch on DEV."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(X.shape[0], (1,), generator=g).item())
    idx = torch.empty(n, dtype=torch.long, device=DEV)
    idx[0] = first
    d = torch.cdist(X[first:first + 1], X)[0]          # (N,)
    for i in range(1, n):
        nxt = int(torch.argmax(d).item())
        idx[i] = nxt
        d = torch.minimum(d, torch.cdist(X[nxt:nxt + 1], X)[0])
    return idx


def knn_max(bank, X):
    q = torch.from_numpy(X.reshape(-1, X.shape[2]).astype("f4")).to(DEV)
    mins = []
    for i in range(0, q.shape[0], 2048):
        mins.append(torch.cdist(q[i:i + 2048], bank).min(1).values)
    out = torch.cat(mins).cpu().numpy().reshape(X.shape[0], X.shape[1]).max(1)
    del q; torch.cuda.empty_cache()
    return out


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
            b = (R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
                 / f"psad_scores_{cat}_seed{s}_hc_tta_merge")
            zc = np.load(f"{b}_test.npz", allow_pickle=True)
            zcv = np.load(f"{b}_val.npz", allow_pickle=True)
            fixed[(cat, s)] = {
                "lt": np.array([str(x) for x in ze["label_type"]]),
                "ead": z(np.asarray(ze["score"], float), np.asarray(zev[zev.files[0]], float)),
                "comp": z(np.asarray(zc["scores"], float), np.asarray(zcv["scores"], float))}

    def ls(score, lt):
        g = lt == "good"
        au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                           np.r_[score[g], score[m]]))
        return 0.5 * (au(lt == "logical") + au(lt == "structural"))

    acc = {m: {"alone": [], "fusion": [], "sec": []} for m in ("random", "greedy")}
    per_cat = {}
    for cat in CATS:
        c = np.load(CACHE / f"{cat}.npz")
        ntr = len(list((LOCO / cat / "train/good").glob("*.png")))
        tr_all = c[f"train_{LAYER}"]
        tr, va = tr_all[:ntr].astype("f4"), tr_all[ntr:].astype("f4")
        te = c[f"test_{LAYER}"].astype("f4")
        fl = torch.from_numpy(tr.reshape(-1, tr.shape[2])).to(DEV)
        pc = {}
        for m in ("random", "greedy"):
            a, fu, sc = [], [], []
            for s in SEEDS:
                t0 = time.perf_counter()
                if m == "random":
                    sel = torch.from_numpy(np.random.default_rng(s).choice(
                        fl.shape[0], min(BANK, fl.shape[0]), replace=False)).to(DEV)
                else:
                    sel = greedy_kcenter(fl, min(BANK, fl.shape[0]), s)
                bank = fl[sel].contiguous()
                torch.cuda.synchronize(); sc.append(time.perf_counter() - t0)
                v, t = knn_max(bank, va), knn_max(bank, te)
                f = fixed[(cat, s)]
                a.append(ls(t, f["lt"]))
                fu.append(ls(f["ead"] + z(t, v) + f["comp"], f["lt"]))
                del bank
            pc[m] = {"alone": float(np.mean(a)), "fusion": float(np.mean(fu)),
                     "build_sec": float(np.mean(sc))}
            acc[m]["alone"] += a; acc[m]["fusion"] += fu; acc[m]["sec"] += sc
            print(f"  [{cat} {m:6s}] PC 단독 {pc[m]['alone']:.4f}  융합 {pc[m]['fusion']:.4f}  "
                  f"뱅크 구축 {pc[m]['build_sec']:.1f}s", flush=True)
        per_cat[cat] = pc
        del fl, c
        torch.cuda.empty_cache()

    agg = {m: {"pc_alone_LS": float(np.mean(acc[m]["alone"])),
               "fusion_LS": float(np.mean(acc[m]["fusion"])),
               "fusion_std": float(np.std([np.mean(acc[m]["fusion"][i::3]) for i in range(3)], ddof=1)),
               "bank_build_sec": float(np.mean(acc[m]["sec"]))} for m in acc}
    d_alone = agg["greedy"]["pc_alone_LS"] - agg["random"]["pc_alone_LS"]
    d_fus = agg["greedy"]["fusion_LS"] - agg["random"]["fusion_LS"]
    out = {"layer": LAYER, "bank": BANK, "seeds": SEEDS, "aggregate": agg, "per_cat": per_cat,
           "delta_greedy_minus_random": {"pc_alone": d_alone, "fusion": d_fus},
           "cost_ratio": agg["greedy"]["bank_build_sec"] / max(agg["random"]["bank_build_sec"], 1e-9),
           "framing": "사후 절제다 — 무작위 표본은 이미 확정된 설정이고 결과를 보고 바꾸지 않는다.",
           "note": "greedy k-center = PatchCore 의 coreset 선택법. 뱅크 크기·k·집계를 동일하게 두고 "
                   "선택법만 바꿨다."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n  {'선택법':10s}{'PC 단독':>10s}{'융합 L+S':>11s}{'융합 std':>10s}{'뱅크 구축':>12s}")
    for m in ("random", "greedy"):
        a = agg[m]
        print(f"  {m:10s}{a['pc_alone_LS']:>10.4f}{a['fusion_LS']:>11.4f}"
              f"{a['fusion_std']:>10.4f}{a['bank_build_sec']:>11.1f}s")
    print(f"\n  greedy − random:  PC 단독 {d_alone:+.4f}   융합 {d_fus:+.4f}   "
          f"구축 비용 {out['cost_ratio']:.0f}배")
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
