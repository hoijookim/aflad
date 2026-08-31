#!/usr/bin/env python3
"""요청 ④ 의 대체 — **패치 메모리 자리**의 분기를 갈아끼워 융합을 다시 채점한다.

왜 이걸 하나
  ④ 가 요청한 SALAD 교체(A~D)는 **재집계로 안 된다.** SALAD 는 test 점수만 있고 val 점수가
  없으며, 체크포인트(reports/phase0/salad_reproduction_multiseed)가 이 머신에 없다.
  z-정규화 기준값을 만들려면 재학습이 필요하고 학습 스크립트가 스스로 ~22h 라고 적고 있다.

  val 점수까지 갖춘 분기는 EAD-S · PC · 구성(hc/hcp) 넷뿐이다. 그래서 지금 답할 수 있는 것은
  **패치 메모리 자리의 교체**다 — "이 조합인 이유" 라는 물음에 한 자리에서나마 답한다.
  ③ 에서 만든 5종 캐시를 그대로 쓴다.

계층 선택 (test 를 보지 않는다)
  주 설정은 DINOv3-L 24블록 중 L17(상대깊이 0.708)이다. 다른 backbone 도 **같은 상대깊이**에
  가장 가까운 보유 계층을 쓴다. 12블록 모델은 L9 다. 성능을 보고 고르지 않는다.

뱅크는 train-only 다 — val 을 채점해야 하므로 자기 패치가 뱅크에 있으면 안 된다
(val_scores_pc.py 와 동일한 이유·동일한 슬라이싱).
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CACHE = R / "cache"
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BANK = 50000
OUT = R / "reports/countgd/pc_backbone_swap.json"

# (표시명, 캐시, 블록수, 쓸 계층) — 계층은 상대깊이 17/24 에 가장 가까운 보유 계층
BACKBONES = [("DINOv3-S/16", "dinov3_multilayer_vits16", 12, "L9"),
             ("DINOv3-B/16", "dinov3_multilayer_vitb16", 12, "L9"),
             ("DINOv3-L/16", "dinov3_multilayer_vitl16", 24, "L17"),   # 주 설정
             ("DINOv2-B/14", "dinov2_multilayer_vitb14", 12, "L9"),
             ("DINOv2-L/14", "dinov2_multilayer_vitl14", 24, "L17")]


def knn_max(bank, X):
    b = torch.from_numpy(bank).to(DEV)
    q = torch.from_numpy(X.reshape(-1, X.shape[2]).astype("f4")).to(DEV)
    mins = []
    for i in range(0, q.shape[0], 2048):
        mins.append(torch.cdist(q[i:i + 2048], b).min(1).values)
    out = torch.cat(mins).cpu().numpy().reshape(X.shape[0], X.shape[1]).max(1)
    del b, q
    torch.cuda.empty_cache()
    return out


def pc_scores(cdir, layer, cat, seed):
    """(val, test) 점수. 캐시의 train 블록 = train_good + val_good (sorted) 이다."""
    z = np.load(CACHE / cdir / f"{cat}.npz")
    ntr = len(list((LOCO / cat / "train/good").glob("*.png")))
    tr_all = z[f"train_{layer}"]
    tr, va = tr_all[:ntr].astype("f4"), tr_all[ntr:].astype("f4")
    te = z[f"test_{layer}"].astype("f4")
    nva = len(list((LOCO / cat / "validation/good").glob("*.png")))
    assert va.shape[0] == nva, f"{cat}/{cdir}: val 개수 불일치 {va.shape[0]} vs {nva}"
    fl = tr.reshape(-1, tr.shape[2])
    if fl.shape[0] > BANK:
        fl = fl[np.random.default_rng(seed).choice(fl.shape[0], BANK, replace=False)]
    return knn_max(fl, va), knn_max(fl, te)


def main():
    z_ = lambda t, v: (t - v.mean()) / (v.std() + 1e-12)
    # 고정 분기(EAD, 구성)를 미리 읽는다
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
                "ead": z_(np.asarray(ze["score"], float), np.asarray(zev[zev.files[0]], float)),
                "comp": z_(np.asarray(zc["scores"], float), np.asarray(zcv["scores"], float))}

    def ls(score, lt):
        g = lt == "good"
        au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                           np.r_[score[g], score[m]]))
        lo, st = au(lt == "logical"), au(lt == "structural")
        return lo, st, 0.5 * (lo + st)

    res = {}
    for name, cdir, nb, layer in BACKBONES:
        per_seed = {}
        for s in SEEDS:
            rows = {}
            for cat in CATS:
                v, t = pc_scores(cdir, layer, cat, s)
                f = fixed[(cat, s)]
                pcz = z_(t, v)
                rows[cat] = {"fusion3": ls(f["ead"] + pcz + f["comp"], f["lt"]),
                             "pc_alone": ls(pcz, f["lt"])}
            per_seed[s] = {k: [float(np.mean([rows[c][k][i] for c in CATS])) for i in range(3)]
                           for k in ("fusion3", "pc_alone")}
            print(f"  [{name} {layer} seed{s}] 융합 L+S {per_seed[s]['fusion3'][2]:.4f}  "
                  f"PC 단독 {per_seed[s]['pc_alone'][2]:.4f}", flush=True)
        agg = {}
        for k in ("fusion3", "pc_alone"):
            arr = np.array([per_seed[s][k] for s in SEEDS])
            agg[k] = {"logical": float(arr[:, 0].mean()), "structural": float(arr[:, 1].mean()),
                      "LS": float(arr[:, 2].mean()), "LS_std": float(arr[:, 2].std(ddof=1))}
        res[name] = {"layer": layer, "n_blocks": nb, **agg,
                     "per_seed_LS": {str(s): per_seed[s]["fusion3"][2] for s in SEEDS}}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"backbones": res, "fixed_branches": "EAD-S + 구성 hc (주 설정과 동일)",
               "rule": "공식 검증셋 정상 표본 z-정규화 + 등가중 합, 시드 42/43/44. "
                       "뱅크는 train-only, coreset 50000, 1-NN, 패치 max.",
               "layer_rule": "주 설정의 상대깊이(24블록 중 L17 = 0.708)에 가장 가까운 보유 계층. "
                             "test 를 보고 고르지 않았다.",
               "why_not_salad": "SALAD 는 val 점수가 없고 체크포인트가 이 머신에 없다. "
                                "재학습이 필요하며 salad_multiseed_train.sh 가 ~22h 로 적고 있다."},
              open(OUT, "w"), ensure_ascii=False, indent=2)

    print(f"\n=== 패치 메모리 자리 교체 (EAD + [PC] + 구성, 시드 42/43/44)")
    print(f"  {'backbone':14s}{'계층':>6s}{'융합 L+S':>11s}{'std':>9s}{'PC 단독':>10s}{'Δ vs 주설정':>13s}")
    base = res["DINOv3-L/16"]["fusion3"]["LS"]
    for name, _, _, layer in BACKBONES:
        r = res[name]
        d = r["fusion3"]["LS"] - base
        tag = "  ← 주 설정" if name == "DINOv3-L/16" else f"{d:+.4f}"
        print(f"  {name:14s}{layer:>6s}{r['fusion3']['LS']:>11.4f}{r['fusion3']['LS_std']:>9.4f}"
              f"{r['pc_alone']['LS']:>10.4f}{tag:>13s}")
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
