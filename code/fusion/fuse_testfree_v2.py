#!/usr/bin/env python3
"""test-free 융합 v2 — 분기 구성 확장 실험.

제약(전부 유지): validation/good 통계로만 z-정규화, 등가중, 전수 test AUROC.
테스트 이미지·라벨은 어떤 단계에서도 쓰지 않는다.

사용 가능한 신호(전부 이미 보유):
  ead      EfficientAD-S, 재학습 5-seed 점수 평균
  pcL17    DINOv3-L L17 PatchCore (논문 구성)
  pcL8     DINOv3-L L8
  pcL23    DINOv3-L L23
  pcv3b    DINOv3-B L11
  pcv2b    DINOv2-B L11
  psad_hc  재구축 구성분기 (히스토그램+임베딩)
  psad_hcp 재구축 구성분기 (+내부 PatchCore)

PatchCore 계열은 뱅크를 train_good 만으로 만들고(val 자기포함 방지), 시드 42-46 의
점수를 평균한다. 결과는 캐시해 재사용한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

R = Path("/workspace/ai-vision-research")
EAD_T = R / "reports/phase0/efficient_ad_official_small/npz"
EAD_V = R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
REB = R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
CACHE_DIR = R / "reports/countgd/testfree_branch_cache"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44, 45, 46]

PC_SPEC = {   # name -> (캐시 디렉토리, 레이어)
    "pcL17": ("cache/dinov3_multilayer_vitl16", 17),
    "pcL8": ("cache/dinov3_multilayer_vitl16", 8),
    "pcL23": ("cache/dinov3_multilayer_vitl16", 23),
    "pcv3b": ("cache/dinov3_multilayer_vitb16", 11),
    "pcv2b": ("cache/dinov2_multilayer_vitb14", 11),
}


def _pc_one(cdir, layer, cat, seed, bank_size=50000):
    c = np.load(R / cdir / f"{cat}.npz")
    Xtr = c[f"train_L{layer}"].astype("f4")
    Xte = c[f"test_L{layer}"].astype("f4")
    n_val = len(list((R / "datasets/MVTecLOCO" / cat / "validation" / "good").glob("*.png")))
    n_tr = Xtr.shape[0] - n_val
    bank_src, val_src = Xtr[:n_tr], Xtr[n_tr:]
    npt, d = Xtr.shape[1], Xtr.shape[2]
    fl = bank_src.reshape(-1, d)
    idx = np.random.default_rng(seed).choice(fl.shape[0], min(bank_size, fl.shape[0]),
                                             replace=False)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(fl[idx])

    def sc(X):
        di, _ = nn.kneighbors(X.reshape(-1, d))
        return di.mean(1).reshape(X.shape[0], npt).max(1)

    return sc(Xte), sc(val_src)


def pc_branch(name, cat):
    """시드 평균 PatchCore 점수 (캐시)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{name}_{cat}.npz"
    if f.exists():
        z = np.load(f)
        return z["test"], z["val"]
    cdir, layer = PC_SPEC[name]
    ts, vs = [], []
    for sd in SEEDS:
        t, v = _pc_one(cdir, layer, cat, sd)
        ts.append(t)
        vs.append(v)
    t, v = np.mean(ts, 0), np.mean(vs, 0)
    np.savez(f, test=t, val=v)
    print(f"    [cache] {name}/{cat}", flush=True)
    return t, v


def ead_branch(cat):
    ts, vs = [], []
    for sd in SEEDS:
        ts.append(np.load(EAD_T / f"scores_{cat}_seed{sd}.npz")["score"].astype(float))
        vs.append(np.load(EAD_V / f"val_good_{cat}_seed{sd}.npz")["scores"].astype(float))
    z = np.load(EAD_T / f"scores_{cat}_seed42.npz", allow_pickle=True)
    return np.mean(ts, 0), np.mean(vs, 0), z["label"].astype(int), z["label_type"].astype(str)


def psad_branch(cat, mtype, seed=42):
    t = np.load(REB / f"psad_scores_{cat}_seed{seed}_{mtype}_test.npz", allow_pickle=True)
    v = np.load(REB / f"psad_scores_{cat}_seed{seed}_{mtype}_val.npz", allow_pickle=True)
    return t["scores"].astype(float), v["scores"].astype(float)


def normalize(t, v, mode="robust"):
    """val-good 통계만으로 정규화 (test 무접촉).

    z      : (t-mean)/std — 논문 val_znorm_fusion 과 동일. 구성분기 점수가 heavy-tail
             이라 test 이상치(최대 5e5, val 은 ~1)가 스케일을 지배하는 문제가 있다.
    robust : (t-median)/IQR — 동일 정보(val)만 쓰면서 이상치에 둔감. 공식 psad.py 가
             마지막에 test min-max 로 눌러 해결하던 것을 test 무접촉으로 대체한다.
    """
    if mode == "z":
        return (t - v.mean()) / max(v.std(), 1e-9)
    med = np.median(v)
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    return (t - med) / max(iqr, 1e-9)


def get_branch(name, cat):
    if name == "ead":
        t, v, _, _ = ead_branch(cat)
        return t, v
    if name in PC_SPEC:
        return pc_branch(name, cat)
    if name.startswith("psad_"):
        return psad_branch(cat, name.split("_", 1)[1])
    raise ValueError(name)


def evaluate(branches, norm="robust"):
    """branches: 이름 리스트 -> 5-cat L+S / logical / structural."""
    per_cat = {}
    for cat in CATS:
        _, _, lab, lt = ead_branch(cat)
        f = 0
        for b in branches:
            t, v = get_branch(b, cat)
            assert len(t) == len(lab), f"{b}/{cat} 길이 {len(t)} != {len(lab)}"
            f = f + normalize(t, v, norm)
        lg = roc_auc_score(lab[(lt == "good") | (lt == "logical")],
                           f[(lt == "good") | (lt == "logical")])
        st = roc_auc_score(lab[(lt == "good") | (lt == "structural")],
                           f[(lt == "good") | (lt == "structural")])
        per_cat[cat] = {"logical": lg, "structural": st, "LS": 0.5 * (lg + st)}
    agg = {k: float(np.mean([per_cat[c][k] for c in CATS]))
           for k in ("logical", "structural", "LS")}
    return agg, per_cat


CONFIGS = {
    "paper_equiv (ead+pcL17+hc)": ["ead", "pcL17", "psad_hc"],
    "paper_equiv_hcp": ["ead", "pcL17", "psad_hcp"],
    "multilayer_PC": ["ead", "pcL8", "pcL17", "pcL23", "psad_hc"],
    "multibackbone_PC": ["ead", "pcL17", "pcv3b", "psad_hc"],
    "multi_all_PC": ["ead", "pcL8", "pcL17", "pcL23", "pcv3b", "pcv2b", "psad_hc"],
    "hc+hcp": ["ead", "pcL17", "psad_hc", "psad_hcp"],
    "full": ["ead", "pcL8", "pcL17", "pcL23", "pcv3b", "pcv2b", "psad_hc", "psad_hcp"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--norms", default="z,robust")
    a = ap.parse_args()
    names = list(CONFIGS) if a.configs == "all" else a.configs.split(";")
    out = {}
    for nm in a.norms.split(","):
        print(f"\n--- 정규화: {nm} ---", flush=True)
        for name in names:
            br = CONFIGS[name]
            agg, per_cat = evaluate(br, nm)
            out[f"{name} [{nm}]"] = {"branches": br, "norm": nm,
                                     "aggregate": agg, "per_cat": per_cat}
            print(f"  {name:34s} L+S {agg['LS']:.4f} "
                  f"(log {agg['logical']:.4f} / str {agg['structural']:.4f})", flush=True)
    op = R / "reports/countgd/testfree_config_search.json"
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {op}")
    best = max(out.items(), key=lambda kv: kv[1]["aggregate"]["LS"])
    print(f"최고: {best[0]} = {best[1]['aggregate']['LS']:.4f} "
          f"(논문 test-free 0.9751 대비 {best[1]['aggregate']['LS']-0.9751:+.4f})")
    print("주의: 구성 선택에 test AUROC 를 봤으므로 이 최고치는 상한(선택 편향) — "
          "채택 시 사전 지정 근거 필요")


if __name__ == "__main__":
    main()
