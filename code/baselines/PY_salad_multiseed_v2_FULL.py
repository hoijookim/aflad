"""Path Y #10: SALAD multi-seed FULL inference (paper-faithful test pipeline).

Replaces broken PY_salad_multiseed_auroc.py (which skipped runtime stats).

For each (seed, cat):
  1. Load 5 checkpoints (teacher/student/ae/comp_ae/comp_unet)
  2. teacher_normalization on train (per-channel mean/std)
  3. map_normalization on val (q_st, q_ae quantiles)
  4. score_normalization on val (q_eff, q_seg quantiles)
  5. extract_features_mahalanobis on train (Maha mean+covinv × 3 variants)
  6. map_normalization_mahalanobis on val (q_start_mah, q_end_mah)
  7. test on test_set — capture per-image score (combined + img + maha + comp)

Output per (seed, cat):
  - reports/path_y/salad_multiseed_v2/{cat}_seed{N}.npz
    (score_combined, score_img, score_maha, score_comp, label_binary, label_type)
  - reports/path_y/salad_multiseed_v2/aggregate.json

Cost: ~5-10 min per (seed, cat). 15 combos = 1.5-2.5h GPU total.
"""
from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

REPO = Path("/workspace/ai-vision-research")
SALAD_DIR = REPO / "external" / "SALAD"
sys.path.insert(0, str(SALAD_DIR))

# Switch CWD so SALAD relative imports work
os.chdir(SALAD_DIR)

import numpy_patch  # noqa: F401
from salad_dataset import (
    ImageFolderWithoutTarget, ImageFolderWithPath,
    ImageFolderWithoutTargetWithSeg, ImageFolderWithPathWithSeg,
)
from train_salad import (
    teacher_normalization, map_normalization, score_normalization,
    extract_features_mahalanobis, map_normalization_mahalanobis,
    predict, predict_mahalanobis, predict_comp_map,
    train_transform, default_transform,
)

LOCO_DIR = REPO / "datasets" / "MVTecLOCO"
SEG_DIR = SALAD_DIR / "data" / "mvtec_loco_composition_maps"
CKPT_BASE = REPO / "reports" / "phase0" / "salad_reproduction_multiseed"
OUT_DIR = REPO / "reports" / "path_y" / "salad_multiseed_v2"

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = ["seed42", "seed43", "seed44"]
DEVICE = "cuda"


def build_loaders(cat):
    full_train_set = ImageFolderWithoutTarget(str(LOCO_DIR / cat / "train"),
        transform=transforms.Lambda(train_transform))
    full_train_seg_set = ImageFolderWithoutTarget(str(SEG_DIR / cat / "train"),
        transform=transforms.Lambda(train_transform))
    full_train_set.seg = False
    full_train_seg_set.seg = True
    full_train_set = ImageFolderWithoutTargetWithSeg(full_train_set, full_train_seg_set)

    val_set = ImageFolderWithoutTarget(str(LOCO_DIR / cat / "validation"),
        transform=transforms.Lambda(train_transform))
    val_seg = ImageFolderWithoutTarget(str(SEG_DIR / cat / "validation"),
        transform=transforms.Lambda(train_transform))
    val_set.seg = False
    val_seg.seg = True
    val_set = ImageFolderWithoutTargetWithSeg(val_set, val_seg)

    test_set = ImageFolderWithPath(str(LOCO_DIR / cat / "test"),
        transform=default_transform)
    test_seg = ImageFolderWithoutTarget(str(SEG_DIR / cat / "test"),
        transform=default_transform)
    test_set.seg = False
    test_seg.seg = True
    test_set = ImageFolderWithPathWithSeg(test_set, test_seg)

    train_loader = DataLoader(full_train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=2)
    return train_loader, full_train_set, val_loader, test_set


def run_one(seed_name, cat):
    print(f"\n=== {seed_name} / {cat} ===", flush=True)
    ckpt_dir = CKPT_BASE / seed_name / cat / cat
    if not (ckpt_dir / "teacher_final.pth").exists():
        print(f"  [SKIP] ckpt missing: {ckpt_dir}", flush=True)
        return None

    teacher = torch.load(ckpt_dir / "teacher_final.pth", map_location=DEVICE, weights_only=False)
    student = torch.load(ckpt_dir / "student_final.pth", map_location=DEVICE, weights_only=False)
    autoencoder = torch.load(ckpt_dir / "autoencoder_final.pth", map_location=DEVICE, weights_only=False)
    comp_ae = torch.load(ckpt_dir / "comp_autoencoder_final.pth", map_location=DEVICE, weights_only=False)
    comp_unet = torch.load(ckpt_dir / "comp_unet_final.pth", map_location=DEVICE, weights_only=False)
    for m in [teacher, student, autoencoder, comp_ae, comp_unet]:
        m.eval().cuda()

    train_loader, train_set, val_loader, test_set = build_loaders(cat)

    t0 = time.time()
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    q_st_s, q_st_e, q_ae_s, q_ae_e = map_normalization(
        val_loader, teacher, student, autoencoder, teacher_mean, teacher_std,
        desc=f'map_norm_{cat}')
    q_eff_s, q_eff_e, q_seg_s, q_seg_e = score_normalization(
        val_loader, teacher, student, autoencoder, comp_ae, comp_unet,
        teacher_mean, teacher_std, q_st_s, q_st_e, q_ae_s, q_ae_e,
        desc=f'score_norm_{cat}')
    (fv_mean, fv_covinv, fv_mean_seg, fv_covinv_seg,
     fv_mean_seg_area, fv_covinv_seg_area) = extract_features_mahalanobis(
        train_loader, train_set, student, teacher_mean, teacher_std)
    q_start_mah, q_end_mah = map_normalization_mahalanobis(
        val_loader, student, teacher_mean, teacher_std,
        fv_covinv, fv_mean, fv_covinv_seg, fv_mean_seg,
        fv_covinv_seg_area, fv_mean_seg_area)
    print(f"  stats: {time.time()-t0:.1f}s", flush=True)

    # Custom test loop to capture per-image scores
    scores_comb = []
    scores_img = []
    scores_maha = []
    scores_comp = []
    labels = []
    label_types = []  # 'good' | 'logical' | 'structural'

    t1 = time.time()
    with torch.no_grad():
        for image, seg, target, path in test_set:
            image = image.unsqueeze(0).cuda()
            seg = seg.cuda()

            map_comb, map_st, map_ae = predict(
                image=image, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_s, q_st_end=q_st_e,
                q_ae_start=q_ae_s, q_ae_end=q_ae_e)
            map_comb_norm = (map_comb - q_eff_s) / q_eff_e
            map_comb_arr = map_comb_norm[0, 0].cpu().numpy()

            map_comp = predict_comp_map(seg, comp_ae, comp_unet).unsqueeze(0)
            map_comp = (map_comp - q_seg_s) / q_seg_e
            map_comp_arr = map_comp[0, 0].cpu().numpy()

            maha_score = predict_mahalanobis(
                image=image, seg=seg, teacher=student,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                feature_vectors_covinv=fv_covinv, feature_vectors_mean=fv_mean,
                feature_vectors_covinv_seg=fv_covinv_seg, feature_vectors_mean_seg=fv_mean_seg,
                feature_vectors_covinv_seg_area=fv_covinv_seg_area,
                feature_vectors_mean_seg_area=fv_mean_seg_area,
                q_start=q_start_mah, q_end=q_end_mah)

            defect = os.path.basename(os.path.dirname(path))
            lbl = 0 if defect == 'good' else 1
            label_types.append(defect if defect == 'good' else
                               'logical' if 'logical' in defect else 'structural')
            labels.append(lbl)
            scores_img.append(float(np.max(map_comb_arr)))
            scores_maha.append(float(maha_score))
            scores_comp.append(float(np.max(map_comp_arr)))
            scores_comb.append(float(np.max(map_comb_arr) + maha_score + np.max(map_comp_arr)))
    n_test = len(labels)
    print(f"  inference: {time.time()-t1:.1f}s ({n_test} images)", flush=True)

    # Compute AUROC breakdowns (Paper: all_auroc = 0.5*(log+str))
    labels = np.array(labels)
    label_types = np.array(label_types)
    scores_comb = np.array(scores_comb)
    scores_img = np.array(scores_img)
    scores_maha = np.array(scores_maha)
    scores_comp = np.array(scores_comp)

    def auc_breakdown(scores):
        good = label_types == 'good'
        log_mask = good | (label_types == 'logical')
        str_mask = good | (label_types == 'structural')
        try:
            log_a = roc_auc_score(labels[log_mask], scores[log_mask])
            str_a = roc_auc_score(labels[str_mask], scores[str_mask])
            return {"logical": float(log_a), "structural": float(str_a),
                    "all": float(0.5 * (log_a + str_a))}
        except Exception:
            return {"logical": float("nan"), "structural": float("nan"), "all": float("nan")}

    result = {
        "ckpt_dir": str(ckpt_dir),
        "n_test": n_test,
        "combined": auc_breakdown(scores_comb),
        "image": auc_breakdown(scores_img),
        "maha": auc_breakdown(scores_maha),
        "comp": auc_breakdown(scores_comp),
    }
    print(f"  AUROC: comb={result['combined']['all']:.4f} img={result['image']['all']:.4f} maha={result['maha']['all']:.4f} comp={result['comp']['all']:.4f}", flush=True)

    # Save per-image scores
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{cat}_{seed_name}.npz",
             combined=scores_comb, image=scores_img,
             maha=scores_maha, comp=scores_comp,
             labels=labels, label_types=label_types)
    return result


def main():
    print(f"PY_salad_multiseed_v2 FULL inference (paper-faithful)\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = {}
        for cat in CATS:
            try:
                r = run_one(seed, cat)
                if r:
                    all_results[seed][cat] = r
                    # Persist incremental to survive crashes
                    (OUT_DIR / "aggregate.json").write_text(json.dumps(all_results, indent=2))
            except Exception as e:
                import traceback
                print(f"  [ERR] {seed}/{cat}: {e}", flush=True)
                traceback.print_exc()

    # 5-cat × 3-seed aggregate per metric
    print("\n========== Aggregate ==========")
    summary = {"per_seed": all_results, "aggregate": {}}
    for metric in ("combined", "image", "maha", "comp"):
        per_seed_aurocs = []
        for seed in SEEDS:
            if seed not in all_results: continue
            cat_aurocs = [r[metric]["all"] for r in all_results[seed].values()]
            if cat_aurocs:
                per_seed_aurocs.append(np.mean(cat_aurocs))
        if per_seed_aurocs:
            arr = np.array(per_seed_aurocs)
            summary["aggregate"][metric] = {
                "per_seed_mean": [float(x) for x in arr],
                "seeds_mean": float(arr.mean()),
                "seeds_std": float(arr.std()),
            }
            print(f"  {metric:<10} {arr.mean():.4f} ± {arr.std():.4f}  (per-seed: {[f'{x:.4f}' for x in arr]})")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] saved {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
