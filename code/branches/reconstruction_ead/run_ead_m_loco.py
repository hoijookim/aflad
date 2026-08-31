"""Phase 0.1 — EfficientAD-M LOCO baseline reproduction.

Roadmap: docs/roadmaps/260525_convergent_design_roadmap.md §3.2.1
Gate 0: 5-cat mean image AUROC >= 90.2% (paper 90.7 - 0.5).

Key choices:
- model_size = "medium" per roadmap (paper EAD-M LOCO ~90.7%)
- Teacher = pretrained_teacher_medium.pth (already cached, ImageNet pretrained)
- Penalty regularizer dir = imagenette (anomalib default; full ImageNet not
  available locally; roadmap §3.4 acknowledges this trade-off)
- 70k iterations, batch=1, lr=1e-4 (paper hyperparams)
- 3 seeds (42, 43, 44) per roadmap convention
- Per-image scores saved as npz for downstream calibration / fusion

Usage:
    # Sanity run (single cat, single seed)
    python3.12 scripts/phase0/run_ead_m_loco.py --cat screw_bag --seeds 42

    # Full Phase 0.1 (5 cat × 3 seed, ~17h on RTX 5090)
    python3.12 scripts/phase0/run_ead_m_loco.py

Output:
    reports/phase0/efficient_ad_medium/
        scores_{cat}_seed{42,43,44}.npz       per-image score + label + label_type
        summary_{cat}_seed{42,43,44}.json     per-run metrics (image AUROC, all/log/str)
        aggregate.json                         5-cat × 3-seed roll-up
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_loco_unified import evaluate_loco, aggregate_cats, CATS_LOCO  # noqa: E402

from anomalib.data.datamodules.image.mvtec_loco import MVTecLOCO
from anomalib.engine import Engine
from anomalib.models import EfficientAd

REPO = Path("/workspace/ai-vision-research")
LOCO_ROOT = REPO / "datasets" / "MVTecLOCO"
# Penalty regularizer image dir. roadmap §3.2.1: ImageNet-1K 권장, Imagenette 금지.
# fall back to imagenette if ImageNet not ready (Phase 0 deviation, documented).
IMAGENET_DIR_DEFAULT = REPO / "datasets" / "imagenet1k" / "train"
IMAGENETTE_FALLBACK = REPO / "datasets" / "imagenette" / "imagenette2"
OUT_ROOT_DEFAULT = REPO / "reports" / "phase0" / "efficient_ad_medium"
SHARED_RESULTS_DIR = Path("/workspace/shared/vision-results")

CATS = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]
DEFAULT_SEEDS = [42, 43, 44]
MAX_STEPS = 70_000
MODEL_SIZE = "medium"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_label_type(image_path: str) -> str:
    p = Path(image_path).as_posix()
    if "/logical_anomalies/" in p:
        return "logical"
    if "/structural_anomalies/" in p:
        return "structural"
    if "/good/" in p:
        return "good"
    return "unknown"


def run_one(cat: str, seed: int, imagenet_dir: Path, out_root: Path) -> dict:
    set_all_seeds(seed)

    model = EfficientAd(
        imagenet_dir=str(imagenet_dir),
        model_size=MODEL_SIZE,
        teacher_out_channels=384,
        lr=1e-4,
        weight_decay=1e-5,
    )

    datamodule = MVTecLOCO(
        root=str(LOCO_ROOT),
        category=cat,
        train_batch_size=1,
        eval_batch_size=32,
        num_workers=8,
        seed=seed,
    )

    work_dir = out_root / f"workdir_{cat}_seed{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(work_dir),
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        max_epochs=-1,
        max_steps=MAX_STEPS,
        check_val_every_n_epoch=1000,
        num_sanity_val_steps=0,
    )

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_secs = time.time() - t0

    t0 = time.time()
    predictions = engine.predict(model=model, datamodule=datamodule)
    infer_secs = time.time() - t0

    scores, labels, label_types = [], [], []
    for batch in predictions or []:
        ps = batch.pred_score.detach().cpu().numpy().reshape(-1)
        gl = batch.gt_label.detach().cpu().numpy().reshape(-1).astype(int)
        paths = batch.image_path
        paths_list = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        for s, g, p in zip(ps.tolist(), gl.tolist(), paths_list):
            scores.append(float(s))
            labels.append(int(g))
            label_types.append(infer_label_type(str(p)))

    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    label_types = np.array(label_types)

    # UNIFIED EVAL: use evaluate_loco from eval_loco_unified
    eval_out = evaluate_loco(scores, labels, label_types)

    metrics: dict = {
        "phase": "0.1",
        "model": f"EfficientAD-{MODEL_SIZE[0].upper()}",
        "model_size": MODEL_SIZE,
        "max_steps": MAX_STEPS,
        "imagenet_dir": str(imagenet_dir),
        "cat": cat,
        "seed": seed,
        "train_secs": round(train_secs, 1),
        "infer_secs": round(infer_secs, 1),
        "n_test": int(len(labels)),
        "timestamp": datetime.now().isoformat(),
        # Flat per-mode metrics for backward-compat readers
        "auroc_all": eval_out["all"]["auroc"],
        "aupr_all": eval_out["all"]["aupr"],
        "f1_max_all": eval_out["all"]["f1_max"],
        "n_all": eval_out["all"]["n"],
        "auroc_logical": eval_out["logical"]["auroc"],
        "aupr_logical": eval_out["logical"]["aupr"],
        "f1_max_logical": eval_out["logical"]["f1_max"],
        "n_logical": eval_out["logical"]["n"],
        "auroc_structural": eval_out["structural"]["auroc"],
        "aupr_structural": eval_out["structural"]["aupr"],
        "f1_max_structural": eval_out["structural"]["f1_max"],
        "n_structural": eval_out["structural"]["n"],
        # Canonical nested form (matches eval_loco_unified)
        "eval": eval_out,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    npz_path = out_root / f"scores_{cat}_seed{seed}.npz"
    np.savez(npz_path, score=scores, label=labels, label_type=label_types)
    metrics_path = out_root / f"summary_{cat}_seed{seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"[OK] {cat}/seed{seed}  AUROC_all={metrics['auroc_all']:.4f}  "
        f"AUROC_log={metrics['auroc_logical']:.4f}  AUROC_str={metrics['auroc_structural']:.4f}  "
        f"(train={train_secs:.0f}s, infer={infer_secs:.0f}s)"
    )
    return metrics


def build_aggregate(results: list[dict]) -> dict:
    """Use eval_loco_unified.aggregate_cats for naming consistency with Phase 0.3 / 1A.

    Output 5cat_mean keys: 'all_auroc', 'logical_auroc', 'structural_auroc', etc.
    (matches eval_loco_unified convention, NOT the legacy 'auroc_all' form)
    """
    per_cat_per_seed: dict[str, list[dict]] = {}
    for r in results:
        per_cat_per_seed.setdefault(r["cat"], []).append(r["eval"])

    agg_core = aggregate_cats(per_cat_per_seed)
    agg = {
        "phase": "0.1",
        "model": f"EfficientAD-{MODEL_SIZE[0].upper()}",
        "gate0_threshold": 0.902,
        "gate0_paper_target": 0.907,
        **agg_core,  # adds 'per_cat' and '5cat_mean' with unified keys
    }
    agg["gate0_pass"] = agg["5cat_mean"].get("all_auroc", 0.0) >= agg["gate0_threshold"]
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat", choices=CATS, default=None, help="Single category (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true", help="Skip runs with existing summary json")
    parser.add_argument(
        "--imagenet-dir", type=Path, default=None,
        help="Penalty regularizer image dir. Defaults to imagenet1k/train if exists, else imagenette.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=OUT_ROOT_DEFAULT,
        help="Output dir for scores/summary/aggregate (default reports/phase0/efficient_ad_medium)",
    )
    args = parser.parse_args()

    # Resolve imagenet dir
    if args.imagenet_dir is not None:
        imagenet_dir = args.imagenet_dir
    elif IMAGENET_DIR_DEFAULT.exists() and any(IMAGENET_DIR_DEFAULT.iterdir()):
        imagenet_dir = IMAGENET_DIR_DEFAULT
    else:
        imagenet_dir = IMAGENETTE_FALLBACK
    if not imagenet_dir.exists():
        raise FileNotFoundError(f"imagenet_dir does not exist: {imagenet_dir}")
    print(f"[Phase 0.1] penalty regularizer dir: {imagenet_dir}")
    if imagenet_dir == IMAGENETTE_FALLBACK:
        print("  ⚠️  Using Imagenette fallback (roadmap §3.2.1 deviation, documented)")

    cats = [args.cat] if args.cat else CATS
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    if args.resume:
        for p in out_root.glob("summary_*_seed*.json"):
            try:
                r = json.loads(p.read_text())
                # Ensure backward-compat: if no 'eval' key, rebuild from npz
                if "eval" not in r:
                    npz_p = out_root / f"scores_{r['cat']}_seed{r['seed']}.npz"
                    if npz_p.exists():
                        d = np.load(npz_p)
                        r["eval"] = evaluate_loco(d["score"], d["label"], d["label_type"])
                all_results.append(r)
            except Exception:
                pass

    completed_keys = {(r["cat"], r["seed"]) for r in all_results}
    total = len(cats) * len(args.seeds)
    idx = 0
    for cat in cats:
        for seed in args.seeds:
            idx += 1
            if (cat, seed) in completed_keys:
                print(f"[skip {idx}/{total}] {cat}/seed{seed} already done")
                continue
            print(f"\n[{idx}/{total}] EAD-M  cat={cat}  seed={seed}  steps={MAX_STEPS}")
            try:
                r = run_one(cat, seed, imagenet_dir=imagenet_dir, out_root=out_root)
                all_results.append(r)
            except Exception as e:
                warnings.warn(f"[{cat}/seed{seed}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        agg = build_aggregate(all_results)
        agg_path = out_root / "aggregate.json"
        agg_path.write_text(json.dumps(agg, indent=2))
        import shutil
        shutil.copy2(agg_path, SHARED_RESULTS_DIR / "phase0_ead_medium_aggregate.json")
        print("\n" + "=" * 70)
        print(f"  Phase 0.1 Aggregate → {agg_path}")
        print("=" * 70)
        m = agg["5cat_mean"]
        print(
            f"  5-cat MEAN  AUROC_all={m.get('all_auroc', 0):.4f}  "
            f"AUROC_log={m.get('logical_auroc', 0):.4f}  "
            f"AUROC_str={m.get('structural_auroc', 0):.4f}"
        )
        print(f"  Gate 0 threshold = {agg['gate0_threshold']:.3f}")
        print(f"  Gate 0 PASS = {agg['gate0_pass']}")


if __name__ == "__main__":
    main()
