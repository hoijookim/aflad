#!/usr/bin/env python3
"""표 C2(readout 교체 실험)를 **진짜 개수축**으로 다시 낸다.

기존 산출(d2d3_multidraw_tables67.py 의 table7)은 screw_bag 에서
`auc(s, lab, lt, "logical")` 를 'cardinality' 로 실었다 — 표 C1 과 같은 결함이다.
근거는 그 스크립트 계열의 주석에 남아 있다:
    "⚠ screw_bag만 순수 cardinality; 나머지는 logical=confounded"
그 전제가 틀렸다. screw_bag 논리 137장 = cardinality 48 + **quantity 65** + MIXED/기타 24 로,
quantity 가 최대 블록이다.

readout 계산부(readouts_one / maha_score)는 원 스크립트에서 **그대로 옮겼다** — 바꾼 것은
개수축 인덱스뿐이다. 비교 가능성을 위해 logical 열도 함께 낸다.
"""
import ast
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CACHE = R / "cache"
LOCO = R / "datasets/MVTecLOCO"
SEEDS = [42, 43, 44, 45, 46]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/tableC2_true_cardinality.json"

_tree = ast.parse((R / "scripts/countgd/footprint_resolved_allcat.py").read_text())
STRAND = next(ast.literal_eval(n.value) for n in _tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "STRAND")


def gpu_nn_scores(Xtr, Xte, seed, cs=50000):        # 원 스크립트 그대로
    npat, dd = Xtr.shape[1], Xtr.shape[2]
    fl = Xtr.reshape(-1, dd).astype("f4")
    if fl.shape[0] > cs:
        fl = fl[np.random.default_rng(seed).choice(fl.shape[0], cs, replace=False)]
    bank = torch.from_numpy(fl).to(DEV)
    q = torch.from_numpy(Xte.reshape(-1, dd).astype("f4")).to(DEV)
    mins = []
    for i in range(0, q.shape[0], 2048):
        mins.append(torch.cdist(q[i:i + 2048], bank).min(1).values)
    del bank, q
    torch.cuda.empty_cache()
    return torch.cat(mins).cpu().numpy().reshape(Xte.shape[0], npat).max(1)


def maha_score(Dtr, Dte, seed, pca_dim=None):       # 원 스크립트 그대로
    mu, sd = Dtr.mean(0), Dtr.std(0) + 1e-9
    Zt, Ze = (Dtr - mu) / sd, (Dte - mu) / sd
    if pca_dim and Zt.shape[1] > pca_dim:
        p = PCA(pca_dim, random_state=seed).fit(Zt); Zt = p.transform(Zt); Ze = p.transform(Ze)
    lw = LedoitWolf().fit(Zt); d = Ze - lw.location_
    return np.einsum("ij,jk,ik->i", d, lw.precision_, d)


def readouts_one(cat, K, seed, scorers):            # 원 스크립트 그대로 (반환만 dict)
    z = np.load(CACHE / "dinov3_multilayer_vitl16" / f"{cat}.npz")
    Xtr = z["train_L17"].astype("f4"); Xte = z["test_L17"].astype("f4")
    npat, dd = Xtr.shape[1], Xtr.shape[2]; GRID = int(np.sqrt(npat)); fl = Xtr.reshape(-1, dd)
    r = {}
    s = gpu_nn_scores(Xtr, Xte, seed); r["patch_memory_NN"] = scorers(s)
    rng = np.random.default_rng(seed)
    km = KMeans(K, random_state=seed, n_init=4).fit(
        fl[rng.choice(fl.shape[0], min(40000, fl.shape[0]), replace=False)])
    Ltr = km.predict(fl).reshape(Xtr.shape[0], npat)
    Lte = km.predict(Xte.reshape(-1, dd)).reshape(Xte.shape[0], npat)
    hist = lambda L: np.stack([np.bincount(L[i], minlength=K)
                               for i in range(L.shape[0])]).astype("f4") / npat
    r["area_histogram"] = scorers(maha_score(hist(Ltr), hist(Lte), seed))

    def ic(L):
        o = np.zeros((L.shape[0], K), "f4")
        for i in range(L.shape[0]):
            g = L[i].reshape(GRID, GRID)
            for k in range(K):
                _, n = ndimage.label(g == k); o[i, k] = n
        return o
    r["instance_count"] = scorers(maha_score(ic(Ltr), ic(Lte), seed))
    r["set_distance"] = scorers(maha_score(
        np.concatenate([Xtr.mean(1), Xtr.std(1)], 1),
        np.concatenate([Xte.mean(1), Xte.std(1)], 1), seed, pca_dim=128))
    return r


def axis_index(cat, lt):
    cfg = {d["pixel_value"]: d["defect_name"]
           for d in json.load(open(LOCO / cat / "defects_config.json"))}
    gt = LOCO / cat / "ground_truth/logical_anomalies"
    subs = []
    for p in sorted((LOCO / cat / "test/logical_anomalies").glob("*.png")):
        d = gt / p.stem
        pv = set()
        if d.is_dir():
            for m in d.glob("*.png"):
                pv |= set(np.unique(np.array(Image.open(m))).tolist())
        n = sorted(cfg[v] for v in pv if v in cfg)
        subs.append(n[0] if len(n) == 1 else "MIXED")
    lp = np.where(lt == "logical")[0]
    assert len(lp) == len(subs), cat
    return {"good": np.where(lt == "good")[0], "structural": np.where(lt == "structural")[0],
            "logical": lp,
            "cardinality": np.array([lp[i] for i, n in enumerate(subs)
                                     if n != "MIXED" and STRAND[cat].get(n) == "cardinality"], int)}


def main():
    ms = lambda v: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
    res, counts = {}, {}
    for cat, K in (("screw_bag", 7), ("pushpins", 26)):
        z = np.load(CACHE / "dinov3_multilayer_vitl16" / f"{cat}.npz")
        lt = np.array([str(x) for x in z["test_ltypes"]])
        idx = axis_index(cat, lt)
        counts[cat] = {k: int(len(v)) for k, v in idx.items()}
        g = idx["good"]

        def scorers(s, idx=idx, g=g):
            out = {}
            for k in ("cardinality", "logical", "structural"):
                pos = idx[k]
                out[k] = float(roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(pos))],
                                             np.r_[s[g], s[pos]]))
            return out

        acc = {}
        for sd in SEEDS:
            rr = readouts_one(cat, K, sd, scorers)
            for ro, v in rr.items():
                for k, x in v.items():
                    acc.setdefault(ro, {}).setdefault(k, []).append(x)
            print(f"[{cat} seed {sd}] " + "  ".join(
                f"{ro}: card {v['cardinality']:.3f} log {v['logical']:.3f} str {v['structural']:.3f}"
                for ro, v in rr.items()), flush=True)
        res[cat] = {ro: {k: ms(v) for k, v in d.items()} for ro, d in acc.items()}

    old = json.load(open(R / "reports/countgd/d2d3_multidraw_tables67.json"))["table7"]
    json.dump({"seeds": SEEDS, "per_cat": res, "axis_image_counts": counts, "previous_table7": old,
               "note": "기존 표 C2 의 'cardinality' 열은 screw_bag 에서 logical 이었다. 여기서는 "
                       "부록 C 의 사전등록 STRAND 로 정의한 진짜 개수축과 logical 을 모두 낸다. "
                       "readout 계산부는 d2d3_multidraw_tables67.py 에서 그대로 옮겼다."},
              open(OUT, "w"), ensure_ascii=False, indent=2)

    for cat in res:
        print(f"\n=== {cat} (개수축 {counts[cat]['cardinality']}장 / 논리 {counts[cat]['logical']}장)")
        print(f"  {'readout':18s}{'cardinality':>14s}{'logical':>14s}{'structural':>14s}{'기존 표 C2':>14s}")
        for ro, v in res[cat].items():
            o = old[cat][ro]["cardinality"][0]
            print(f"  {ro:18s}{v['cardinality'][0]:>14.4f}{v['logical'][0]:>14.4f}"
                  f"{v['structural'][0]:>14.4f}{o:>14.4f}")
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
