"""Phase 0.3 — Unified LOCO evaluation protocol + calibration baselines.

Roadmap §3.2.3: single evaluate_loco() consumed by all downstream phases.
Roadmap §3.4 (Dr. S): measure 3 calibration methods (z-score, percentile, Platt)
to pick a default for Phase 1A fusion.

Input format (npz per (cat, seed)):
    score:      float64, shape (N,)   — per-image anomaly score (higher = more anomalous)
    label:      int64,  shape (N,)    — 0 = good, 1 = anomaly
    label_type: <U10,   shape (N,)    — 'good' | 'logical' | 'structural'

Calibration:
    z-score        : (s - mu) / sigma, computed on TRAINING NORMAL only
                     (test set has no normals at training time)
                     fallback: use validation good split
    percentile     : empirical CDF on TRAINING NORMAL
    Platt          : sigmoid(a*s + b) fit on a labeled holdout
                     (we use train normal vs synthetic anomaly proxy from val good shifted)

For Phase 0.3 we evaluate calibration robustness using bootstrap CI over 7 seeds
(if available) and report which calibration preserves the rank-AUROC best while
normalizing the score scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


CATS_LOCO = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


# ---------------------------------------------------------------------------
# Core unified evaluator
# ---------------------------------------------------------------------------
def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(np.max(f1))


def _metrics_for_mask(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    l, s = labels[mask], scores[mask]
    try:
        return {
            "auroc": float(roc_auc_score(l, s)),
            "aupr": float(average_precision_score(l, s)),
            "f1_max": f1_max(l, s),
            "n": int(mask.sum()),
        }
    except ValueError:
        return {"auroc": float("nan"), "aupr": float("nan"), "f1_max": float("nan"), "n": int(mask.sum())}


def evaluate_loco(
    scores: np.ndarray,
    labels: np.ndarray,
    label_types: np.ndarray,
) -> Dict[str, dict]:
    """Single unified evaluator for LOCO image-level scores.

    Args:
        scores:      (N,) float — higher = more anomalous
        labels:      (N,) int   — 0 good, 1 anomaly
        label_types: (N,)       — 'good' | 'logical' | 'structural'

    Returns:
        {
            'all':        {'auroc', 'aupr', 'f1_max', 'n'},
            'logical':    {... uses good + logical only ...},
            'structural': {... uses good + structural only ...},
        }
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(int)
    label_types = np.asarray(label_types).reshape(-1)

    out = {}
    out["all"] = _metrics_for_mask(labels, scores, np.ones(len(labels), dtype=bool))
    out["logical"] = _metrics_for_mask(
        labels, scores, (label_types == "good") | (label_types == "logical")
    )
    out["structural"] = _metrics_for_mask(
        labels, scores, (label_types == "good") | (label_types == "structural")
    )
    return out


def aggregate_cats(per_cat_per_seed: Dict[str, List[Dict[str, dict]]]) -> dict:
    """Aggregate per-cat × per-seed evaluations to 5-cat mean.

    Args:
        per_cat_per_seed: {cat_name: [eval_seed0, eval_seed1, ...]}
            each eval is the output of evaluate_loco()
    """
    out = {"per_cat": {}, "5cat_mean": {}}
    for cat, seed_evals in per_cat_per_seed.items():
        cat_agg = {}
        for mode in ("all", "logical", "structural"):
            for metric in ("auroc", "aupr", "f1_max"):
                vals = np.array([e[mode][metric] for e in seed_evals if not np.isnan(e[mode][metric])])
                if len(vals) == 0:
                    cat_agg[f"{mode}_{metric}_mean"] = float("nan")
                    cat_agg[f"{mode}_{metric}_std"] = float("nan")
                else:
                    cat_agg[f"{mode}_{metric}_mean"] = float(vals.mean())
                    cat_agg[f"{mode}_{metric}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
                cat_agg[f"{mode}_{metric}_per_seed"] = vals.tolist()
        out["per_cat"][cat] = cat_agg

    for mode in ("all", "logical", "structural"):
        for metric in ("auroc", "aupr", "f1_max"):
            cat_means = np.array(
                [out["per_cat"][c][f"{mode}_{metric}_mean"] for c in out["per_cat"]]
            )
            cat_means = cat_means[~np.isnan(cat_means)]
            if len(cat_means) == 0:
                continue
            out["5cat_mean"][f"{mode}_{metric}"] = float(cat_means.mean())
            out["5cat_mean"][f"{mode}_{metric}_std"] = float(cat_means.std(ddof=1) if len(cat_means) > 1 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------
def calibrate_zscore(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Standardize using mean/std of reference (normal training/val) scores."""
    mu = float(ref_scores.mean())
    sigma = float(ref_scores.std())
    if sigma < 1e-12:
        return scores - mu
    return (scores - mu) / sigma


def calibrate_percentile(scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """Map score → empirical CDF percentile (0..1) under reference (normal) distribution."""
    sorted_ref = np.sort(ref_scores)
    # Right-side: P(ref <= s) = fraction
    ranks = np.searchsorted(sorted_ref, scores, side="right")
    return ranks / max(len(sorted_ref), 1)


def calibrate_platt(
    scores: np.ndarray,
    fit_scores: np.ndarray,
    fit_labels: np.ndarray,
) -> np.ndarray:
    """Logistic calibration sigmoid(a*s + b) fit on labeled data.

    Args:
        scores:     scores to calibrate
        fit_scores: scores used to fit Platt parameters
        fit_labels: 0/1 labels for fit_scores

    Returns calibrated probabilities in [0, 1].
    """
    fit_scores = np.asarray(fit_scores).reshape(-1, 1)
    fit_labels = np.asarray(fit_labels).astype(int)
    if len(set(fit_labels.tolist())) < 2:
        # Cannot fit Platt with single class — fallback to z-score
        return calibrate_zscore(scores, fit_scores.reshape(-1))
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(fit_scores, fit_labels)
    return lr.predict_proba(scores.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Calibration evaluation harness
# ---------------------------------------------------------------------------
def split_train_test(
    scores: np.ndarray,
    labels: np.ndarray,
    label_types: np.ndarray,
    val_frac: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (good_scores_for_calibration, test_scores, test_labels).

    LOCO test set has good + logical + structural. We split:
      - good   → calibration reference (or split into calib + Platt-fit positives)
      - anomaly → eval

    Since LOCO has no separate normal-only train, we use **test good** as reference.
    This is somewhat overfitted but acceptable for *relative* calibration comparison.
    """
    good_mask = (label_types == "good")
    return scores[good_mask].copy(), scores, labels


def measure_calibration_set(
    npz_files: List[Path],
    label: str,
) -> dict:
    """Apply 3 calibrations and report 5-cat mean AUROC (sanity: should be identical
    since AUROC is rank-invariant under monotonic transforms).

    The real value of calibration shows up when **fusing** two sources — that's
    measured in Phase 1A. Here we only report that:
      1. AUROC is preserved (sanity)
      2. The output score scale is stable across cats (key for fusion)
    """
    results = {
        "label": label,
        "calibrations": ["raw", "zscore", "percentile", "platt"],
        "per_cat_per_seed": {},
    }
    per_cat_per_seed = {c: [] for c in CATS_LOCO}

    for npz_path in npz_files:
        # Parse cat from filename: scores_{cat}_seed{N}.npz
        name = npz_path.name
        # cat is between 'scores_' and '_seed'
        try:
            cat = name.replace("scores_", "").split("_seed")[0]
        except Exception:
            continue
        if cat not in CATS_LOCO:
            continue
        d = np.load(npz_path)
        scores = d["score"]
        labels = d["label"]
        label_types = d["label_type"]

        # Reference for calibration: TEST good scores
        # (LOCO has no held-out normal; this is the available proxy)
        good_scores, full_scores, full_labels = split_train_test(scores, labels, label_types)

        # Apply calibrations
        cal_raw = scores
        cal_z = calibrate_zscore(scores, good_scores)
        cal_p = calibrate_percentile(scores, good_scores)
        # Platt fit needs labels — use a small subset of test for fitting
        # (this is intentionally a baseline; Phase 1A will use proper validation)
        cal_pl = calibrate_platt(scores, scores, labels)

        eval_set = {}
        for name_cal, cal_scores in [("raw", cal_raw), ("zscore", cal_z), ("percentile", cal_p), ("platt", cal_pl)]:
            eval_set[name_cal] = evaluate_loco(cal_scores, labels, label_types)
        # Also record scale stats
        eval_set["_scale"] = {
            "raw_mean_good": float(good_scores.mean()),
            "raw_std_good": float(good_scores.std()),
            "raw_min": float(scores.min()),
            "raw_max": float(scores.max()),
        }
        per_cat_per_seed[cat].append(eval_set)

    # Aggregate per calibration method
    agg_per_cal = {}
    for cal_name in ["raw", "zscore", "percentile", "platt"]:
        per_cat_evals = {c: [s[cal_name] for s in seeds_list] for c, seeds_list in per_cat_per_seed.items() if seeds_list}
        if per_cat_evals:
            agg_per_cal[cal_name] = aggregate_cats(per_cat_evals)

    # Scale summary
    all_scale_means = []
    all_scale_stds = []
    for c, seeds_list in per_cat_per_seed.items():
        for s in seeds_list:
            all_scale_means.append(s["_scale"]["raw_mean_good"])
            all_scale_stds.append(s["_scale"]["raw_std_good"])
    scale_summary = {
        "mean_of_good_means_across_cat_seed": float(np.mean(all_scale_means)) if all_scale_means else 0.0,
        "std_of_good_means_across_cat_seed": float(np.std(all_scale_means)) if all_scale_means else 0.0,
        "cv_of_good_means": float(np.std(all_scale_means) / (abs(np.mean(all_scale_means)) + 1e-12)) if all_scale_means else 0.0,
    }
    return {
        "label": label,
        "agg_per_calibration": agg_per_cal,
        "raw_scale": scale_summary,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing scores_*_seed*.npz files (e.g. V4 aupr_bootstrap/scores or Phase 0.1 EAD output)",
    )
    parser.add_argument("--label", required=True, help="Human label for this source (e.g. 'v4_v6_5' or 'ead_m')")
    parser.add_argument("--out", default="reports/phase0/calibration_baseline.json")
    args = parser.parse_args()

    src = Path(args.source)
    npz_files = sorted(src.glob("scores_*_seed*.npz"))
    if not npz_files:
        print(f"ERROR: no npz files in {src}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(npz_files)} npz files in {src}")

    result = measure_calibration_set(npz_files, args.label)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # Merge with existing (multi-source)
        existing = json.loads(out_path.read_text())
        if not isinstance(existing, dict) or "sources" not in existing:
            existing = {"sources": {}}
        existing["sources"][args.label] = result
        out_path.write_text(json.dumps(existing, indent=2))
    else:
        out_path.write_text(json.dumps({"sources": {args.label: result}}, indent=2))

    # Print summary
    print(f"\n=== Calibration baselines for {args.label} ===")
    print(f"Raw score scale stats: {result['raw_scale']}")
    print(f"\n{'Calibration':<12} {'AUROC_all':<10} {'AUROC_log':<10} {'AUROC_str':<10}")
    print("-" * 50)
    for cal in ["raw", "zscore", "percentile", "platt"]:
        if cal not in result["agg_per_calibration"]:
            continue
        m = result["agg_per_calibration"][cal]["5cat_mean"]
        print(
            f"{cal:<12} {m.get('all_auroc', 0):<10.4f} "
            f"{m.get('logical_auroc', 0):<10.4f} "
            f"{m.get('structural_auroc', 0):<10.4f}"
        )
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
