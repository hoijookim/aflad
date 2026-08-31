"""PatchCore family (DINOv2-B / DINOv3-B / DINOv3-L) 를 5-seed {42-46} 로 확장.

배경: PY_ls_patchcore_family.py 는 pc_seed=42 단일 시드였다(표 5의 DINOv3-B/DINOv2-B 행 n_seed=1).
본 스크립트는 코어셋 뱅크 시드만 42..46 으로 바꿔 mean±std(ddof=1) 를 산출한다.
patchcore_score() 는 원본과 동일한 구현(cs=50000, k=1, agg=max)을 그대로 복제했다.

교차검증: pc_seed=42 단독 값이 PY_ls_patchcore_family.py 산출(ls_canonical_scores.json)과
일치해야 한다 — 일치하면 5-seed 확장이 같은 계산 위에 있음이 보장된다.

산출: reports/path_y/metric_unify/ls_patchcore_family_5seed.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

REPO = "/workspace/ai-vision-research"
os.chdir(REPO)
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44, 45, 46]

_V4 = {c: np.load(f"algml_v6_5/aupr_bootstrap/scores/scores_{c}_seed42.npz", allow_pickle=True)
       for c in CATS}
GT_LABEL = {c: _V4[c]["label"].astype(int) for c in CATS}
GT_LTYPE = {c: _V4[c]["label_type"].astype(str) for c in CATS}

MODELS = {
    "DINOv2-B": ("cache/dinov2_multilayer_vitb14", 11),
    "DINOv3-B": ("cache/dinov3_multilayer_vitb16", 11),
    "DINOv3-L": ("cache/dinov3_multilayer_vitl16", 17),
}
CFG = dict(cs=50000, k=1, agg="max")


def patchcore_score(cache, layer, cs, k, agg, pc_seed):
    """PY_ls_patchcore_family.py 의 동명 함수와 동일 구현."""
    X_tr = cache[f"train_L{layer}"].astype(np.float32)
    X_te = cache[f"test_L{layer}"].astype(np.float32)
    d = X_tr.shape[2]
    flat_tr = X_tr.reshape(-1, d)
    if cs < flat_tr.shape[0]:
        rng = np.random.default_rng(pc_seed)
        idx = rng.choice(flat_tr.shape[0], cs, replace=False)
        bank = flat_tr[idx]
    else:
        bank = flat_tr
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nbrs.fit(bank)
    flat_te = X_te.reshape(-1, d)
    dists, _ = nbrs.kneighbors(flat_te)
    ps = dists.mean(axis=1).reshape(X_te.shape[0], X_te.shape[1])
    return ps.max(axis=1) if agg == "max" else ps.mean(axis=1)


def trio(score, label, ltype):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    ltype = np.asarray(ltype, str)
    pooled = roc_auc_score(label, score)
    lm = (ltype == "good") | (ltype == "logical")
    sm = (ltype == "good") | (ltype == "structural")
    lg = roc_auc_score(label[lm], score[lm])
    st = roc_auc_score(label[sm], score[sm])
    return pooled, lg, st, 0.5 * (lg + st)


def main():
    ref_path = "reports/path_y/metric_unify/ls_canonical_scores.json"
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}

    results = {}
    for name, (cdir, layer) in MODELS.items():
        if not os.path.isdir(cdir):
            print(f"[skip] {name}: 캐시 없음 ({cdir})", flush=True)
            continue
        print(f"\n=== {name}  (L{layer}, {cdir}) ===", flush=True)
        per_seed = {}
        for seed in SEEDS:
            per_cat = {}
            for c in CATS:
                cache = np.load(f"{cdir}/{c}.npz")
                ltype = (cache["test_ltypes"].astype(str)
                         if "test_ltypes" in cache.files else GT_LTYPE[c])
                score = patchcore_score(cache, layer, pc_seed=seed, **CFG)
                assert len(score) == len(GT_LABEL[c]), f"{name}/{c}: 길이 불일치"
                p, lg, st, ls = trio(score, GT_LABEL[c], ltype)
                per_cat[c] = dict(pooled=p, logical=lg, structural=st, LS=ls)
            agg = {k: float(np.mean([per_cat[c][k] for c in CATS]))
                   for k in ("pooled", "logical", "structural", "LS")}
            per_seed[seed] = dict(per_cat=per_cat, aggregate=agg)
            print(f"  seed {seed}: L+S {agg['LS']:.4f}  "
                  f"(log {agg['logical']:.4f} / str {agg['structural']:.4f})", flush=True)

        stat = {}
        for k in ("pooled", "logical", "structural", "LS"):
            vals = np.array([per_seed[s]["aggregate"][k] for s in SEEDS])
            stat[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
        percat_stat = {}
        for c in CATS:
            vals = np.array([per_seed[s]["per_cat"][c]["LS"] for s in SEEDS])
            percat_stat[c] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

        results[name] = {"per_seed": {str(s): per_seed[s]["aggregate"] for s in SEEDS},
                         "aggregate_5seed": stat, "per_cat_LS_5seed": percat_stat,
                         "config": dict(CFG, layer=layer, seeds=SEEDS)}
        print(f"  >> 5-seed L+S = {stat['LS']['mean']:.4f} +- {stat['LS']['std']:.4f}", flush=True)

        # 교차검증: seed42 단독이 기존 1-seed 정본과 일치하는가
        if name in ref:
            old = ref[name]["aggregate"]["LS"]
            new42 = per_seed[42]["aggregate"]["LS"]
            d = abs(old - new42)
            tag = "IDENTICAL" if d == 0 else ("MATCH" if d < 1e-6 else "DIFF")
            print(f"  >> seed42 교차검증: 정본 {old:.6f} vs 재계산 {new42:.6f}  Δ{d:.2e}  {tag}",
                  flush=True)
            results[name]["crosscheck_seed42"] = {"canonical": old, "recomputed": new42,
                                                  "delta": d, "verdict": tag}

    os.makedirs("reports/path_y/metric_unify", exist_ok=True)
    out = "reports/path_y/metric_unify/ls_patchcore_family_5seed.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[saved] {out}")

    print("\n=== 표 5 갱신용 요약 ===")
    for name, r in results.items():
        a = r["aggregate_5seed"]
        print(f"  {name:10s} L+S {a['LS']['mean']:.4f} ± {a['LS']['std']:.4f}  "
              f"| logical {a['logical']['mean']:.4f} ± {a['logical']['std']:.4f}  "
              f"| structural {a['structural']['mean']:.4f} ± {a['structural']['std']:.4f}")


if __name__ == "__main__":
    main()
