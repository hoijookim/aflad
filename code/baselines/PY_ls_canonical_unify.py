"""Re-aggregate ALL baselines onto the SAME metric axis (LOCO L+S canonical).

Audit 260604: comparative study mixed pooled image-AUROC (our internals) vs
L+S canonical (paper-cite anchors). This script recomputes, from cached
per-image scores, BOTH:
  - pooled  = roc_auc_score(label, score)                    (good vs all)
  - L+S     = 0.5*(AUROC(good+logical) + AUROC(good+structural))   (official LOCO)
per category, averaged over available seeds, then mean over 5 cats.

NO retraining, NO re-inference. Pure re-aggregation of cached scores.
Cross-check: V4 L+S must reproduce ~0.9075, SALAD ~0.9469 (known values).

label_type handling:
  - V4 / EAD-S / EAD-M / SALAD store label_type(s) directly.
  - PUAD-S / PUAD-M store only binary labels -> transfer V4's label_type
    after verifying binary labels are element-wise identical (same order).
"""
import numpy as np, glob, os, json
from sklearn.metrics import roc_auc_score

REPO = "/workspace/ai-vision-research"
os.chdir(REPO)
CATS = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors']

# ---- ground-truth label_type per cat (from V4 npz, which stores it) ----
_V4 = {c: np.load(f"algml_v6_5/aupr_bootstrap/scores/scores_{c}_seed42.npz", allow_pickle=True) for c in CATS}
GT_LABEL = {c: _V4[c]['label'].astype(int) for c in CATS}
GT_LTYPE = {c: _V4[c]['label_type'].astype(str) for c in CATS}

def trio(score, label, ltype):
    score = np.asarray(score, float); label = np.asarray(label, int); ltype = np.asarray(ltype, str)
    pooled = roc_auc_score(label, score)
    lm = (ltype == 'good') | (ltype == 'logical')
    sm = (ltype == 'good') | (ltype == 'structural')
    log = roc_auc_score(label[lm], score[lm])
    st  = roc_auc_score(label[sm], score[sm])
    return pooled, log, st, 0.5 * (log + st)

# method -> (file_template_glob, score_key, has_own_label_type, label_key, ltype_key)
METHODS = {
    "V4_DLfree": ("algml_v6_5/aupr_bootstrap/scores/scores_{c}_seed*.npz", "score", True, "label", "label_type"),
    "EAD_S":     ("reports/phase0/efficient_ad_official_small/npz/scores_{c}_seed*.npz", "score", True, "label", "label_type"),
    "EAD_M":     ("reports/phase0/efficient_ad_medium/scores_{c}_seed*.npz", "score", None, "label", "label_type"),
    "PUAD_S":    ("reports/path_y/puad_scores/{c}_puad_seed*.npz", "puad", False, "labels", None),
    "PUAD_M":    ("reports/path_y/puad_m_multiseed_scores/{c}_seed*.npz", "puad", False, "labels", None),
    "SALAD":     ("reports/path_y/salad_multiseed_v2/{c}_seed*.npz", "combined", True, "labels", "label_types"),
}

results = {}
for m, (tmpl, skey, has_lt, lkey, ltkey) in METHODS.items():
    per_cat = {}
    seeds_seen = set()
    for c in CATS:
        files = sorted(glob.glob(tmpl.format(c=c)))
        if not files:
            per_cat[c] = None; continue
        vals = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            if skey not in d.files:  # fallback: pick the score-like key
                cand = [k for k in d.files if k not in (lkey, ltkey, 'label', 'labels', 'label_type', 'label_types')]
                skey_eff = cand[0]
            else:
                skey_eff = skey
            score = np.asarray(d[skey_eff], float)
            label = np.asarray(d[lkey], int)
            # resolve label_type
            if has_lt and ltkey in d.files:
                ltype = np.asarray(d[ltkey], str)
            else:
                assert np.array_equal(label, GT_LABEL[c]), f"{m}/{c}: label order mismatch -> cannot transfer label_type"
                ltype = GT_LTYPE[c]
            vals.append(trio(score, label, ltype))
            seeds_seen.add(os.path.basename(f))
        vals = np.array(vals)  # (nseed, 4)
        per_cat[c] = {"n_seed": len(files), "pooled": vals[:,0].mean(),
                      "logical": vals[:,1].mean(), "structural": vals[:,2].mean(), "LS": vals[:,3].mean()}
    valid = [c for c in CATS if per_cat[c]]
    agg = {k: float(np.mean([per_cat[c][k] for c in valid])) for k in ("pooled","logical","structural","LS")}
    agg["n_cat"] = len(valid); agg["n_seed_files"] = len(seeds_seen)
    results[m] = {"per_cat": {c: (per_cat[c] and {k: float(v) for k,v in per_cat[c].items()}) for c in CATS}, "aggregate": agg}

# ---- report ----
print(f"{'Method':<12}{'pooled':>9}{'L+S':>9}{'logical':>9}{'struct':>9}{'Δ(p-LS)':>9}  cats/seeds")
for m, r in results.items():
    a = r["aggregate"]
    print(f"{m:<12}{a['pooled']:>9.4f}{a['LS']:>9.4f}{a['logical']:>9.4f}{a['structural']:>9.4f}{a['pooled']-a['LS']:>9.4f}  {a['n_cat']}/{a['n_seed_files']}")

# ---- cross-check ----
print("\n=== CROSS-CHECK (must reproduce known values) ===")
v4ls = results["V4_DLfree"]["aggregate"]["LS"]; v4p = results["V4_DLfree"]["aggregate"]["pooled"]
slls = results["SALAD"]["aggregate"]["LS"]
print(f"V4   L+S={v4ls:.4f} (expect ~0.9075), pooled={v4p:.4f} (expect ~0.9077)  -> {'OK' if abs(v4ls-0.9075)<0.003 else 'CHECK'}")
print(f"SALAD L+S={slls:.4f} (expect ~0.9469)                                  -> {'OK' if abs(slls-0.9469)<0.004 else 'CHECK'}")

os.makedirs("reports/path_y/metric_unify", exist_ok=True)
json.dump(results, open("reports/path_y/metric_unify/ls_canonical_scores.json","w"), indent=2)
print("\nsaved -> reports/path_y/metric_unify/ls_canonical_scores.json")
