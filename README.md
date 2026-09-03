# AFLAD — Reproduction Package

Annotation-Free signal fusion for **Logical and structural Anomaly Detection** on MVTec LOCO AD.

This repository exists so that the numbers in the paper can be **recomputed**, not taken on
trust. It ships the per-image anomaly scores that the reported results are built from.

**Main setting — image-level AUROC (L+S) = 0.9723 ± 0.0011** (seeds 42/43/44)
logical 0.9685 · structural 0.9762 · **+2.54 pp** over the same-harness SALAD reproduction (0.9469)

---

## 1. Verify in 30 seconds

The headline is recomputed from per-image scores alone. No 18 GB feature cache, no 9 GB of
checkpoints, no GPU — `numpy` and `scikit-learn` suffice.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```

> **Windows**: the JSON artifacts contain Korean text, so run under UTF-8 —
> `set PYTHONUTF8=1` (or `python -X utf8 ...`). The scripts pass
> `encoding="utf-8"` explicitly, so this is a safety net rather than a requirement.
> File hashes in `MANIFEST.md` are computed on **LF** line endings; `.gitattributes`
> pins that, so a Windows clone with `core.autocrlf=true` still matches.

```
  seed 42: L+S 0.9721 (logical 0.9662 / structural 0.9780)
  seed 43: L+S 0.9714 (logical 0.9682 / structural 0.9745)
  seed 44: L+S 0.9735 (logical 0.9710 / structural 0.9759)

  3-seed L+S = 0.9723 ± 0.0011
    LS         reproduced 0.9723  paper 0.9723  diff 0.00001  PASS
    logical    reproduced 0.9685  paper 0.9685  diff 0.00005  PASS
    structural reproduced 0.9762  paper 0.9762  diff 0.00004  PASS
```

That script *is* the entire test-free protocol: z-normalise each branch score using the
mean and standard deviation of the **normal samples of the official validation split**,
sum with **equal weights (1,1,1)**, compute AUROC per defect type, and report
`L+S = (logical + structural) / 2`. No log scaling, no robust statistics, no clipping.

### Why ship scores rather than code alone

Re-training with the same code, the same pseudo-labels and the same seed **does not
reproduce the branch scores.** We measured this with a replicate run (same machine,
seed 42, independent re-training):

| what differs | Spearman |
|---|--:|
| seed only | 0.9381 |
| machine only | 0.9378 |
| **run only (nothing else)** | **0.9388** |

Changing nothing at all produces the same divergence as changing the seed and the machine.
The cause is `cudnn.benchmark = True` in `train_normal_unet.py`: convolution algorithms are
selected by wall-clock timing, and many of the selected kernels use `atomicAdd` in the
backward pass. The seed fixes **initialisation** — weights of a module that never runs are
bit-identical across runs — but it does not fix the **arithmetic**.

**The fused number does reproduce.** A branch-level gap of −0.0026 shrinks to −0.0007 after
fusion, which is 0.61× the seed standard deviation (0.0011). So the precise reproducibility
claim of this method is *"the fused number reproduces"*, not *"the branch scores reproduce"* —
and checking that requires the score files themselves.

Evidence: `results/tables/psad_variance_decomp.json`, `results/tables/psad_3way_compare.json`.

---

## 2. Method

A sum of three branch scores. Each catches a different failure mode.

| Branch | What it measures | Code |
|---|---|---|
| **Reconstruction** — EfficientAD-S | local texture / structure deviation | `code/branches/reconstruction_ead/` |
| **Patch memory** — DINOv3-L L17 PatchCore | distance from the normal patch manifold | `code/branches/patch_memory_pc/` |
| **Composition** — PSAD-hc | component area histogram (h) + appearance embedding (c) | `code/branches/composition_psad/` |

Component labels for the composition branch come from a U-Net segmenter distilled from CSAD
pseudo-labels, with **two improvements**, both adopted on train/val evidence only:

- **Automatic merging of confusable class pairs** — pairs with
  `C[i,j] + C[j,i] > 0.10` in the confusion matrix between train pseudo-labels and U-Net
  predictions are merged. Across all five categories only `screw_bag c4↔c5` qualifies
  (0.168 / 0.261 / 0.142 over three seeds; the runner-up is ≤ 0.012 everywhere — a **10×**
  margin, so a single threshold isolates that pair without per-category hand-tuning).
- **Rotation TTA** — inference at the same angles used for training augmentation, averaged.
  The rule "must reduce both the mean and the maximum of the validation-normal score tail"
  was applied identically to all five categories; only `screw_bag` passed.

---

## 3. Paper element → artifact → code

| Paper element | Artifact | Producing code |
|---|---|---|
| Main headline 0.9723 | `results/main/testfree_final_eadfix_3seed.json` | `code/fusion/prereg_testfree.py --protocol paper --kind hc --psad-tag _tta_merge` |
| Legacy-protocol variant 0.9733 | `results/main/testfree_final_eadfix_origproto_3seed.json` | same script (seed-averaged EAD·PC) |
| hcp variant (appendix) | `results/main/testfree_final_hcp_3seed.json` | same script, `--kind hcp --psad-tag _tta_merge_k5` |
| Baseline comparison table | `results/tables/table5_3seed_42_43_44.json` | `code/baselines/realign_*.py` |
| PatchCore family (train-only bank) | `results/tables/patchcore_family_trainonly_3seed.json` | `code/branches/patch_memory_pc/patchcore_family_trainonly.py` |
| EAD-M with full ImageNet | `results/tables/realign_eadm_fullimagenet_3seed.json` | `code/baselines/realign_eadm_fullimagenet.py` |
| Branch-alone / ablation | `results/tables/testfree_supplements_3seed.json` | `code/analysis/testfree_supplements.py` |
| Deployment configurations | `results/tables/deployment_operating_point.json` | `code/analysis/deployment_operating_point.py` |
| Requirement–cost curve | `results/tables/requirement_cost_curve.json` | `code/analysis/requirement_cost_curve.py` |
| Subtype × four score sources | `results/tables/subtype_branch_auroc.json` | `code/analysis/subtype_branch_auroc.py` |
| ComAD per-axis seed spread | `results/tables/comad_axis_perseed.json` | `code/analysis/comad_axis_perseed.py` |
| Inference parameter counts | `results/tables/param_counts_inference.json` | `code/analysis/measure_param_inference.py` |
| Forward-execution measurement | `results/tables/forward_profile.json` | `code/analysis/profile_forward_modules.py` |
| EAD-S 5-seed reproduction check | `results/reproduction/ead_5seed_verdict.json` | `code/audit/ead_5seed_verdict.py` |
| Pre-registration of design choices | `results/diagnostics/PREREG_branch_improvement.md` | — |
| Accept / reject evidence | `results/diagnostics/criterion_*.json`, `sweep_*.json` | `code/selection_criteria/` |

Directory roles and seed coverage for the per-image scores: `results/per_image_scores/README.md`.

---

## 4. What "test-free" means here

**No configuration was ever selected using the test split.** Decisions used two criteria only.

| Criterion | What it measures | Data |
|---|---|---|
| train discriminability | can the branch separate logical anomalies from normals | perturbations over train pseudo-labels + leave-one-out 1-NN AUROC |
| val stability | does z-normalisation survive outliers | max \|z_robust\| of the validation-normal score tail |

**The two can disagree.** A single validation-normal image that inflates the standard deviation
shrinks that branch's test z-scores across the board, effectively switching it off. Two
candidates were rejected for exactly this reason (k-NN with k=5; a cardinality distance).

**The highest-scoring candidate was also rejected.** An h:c weighting (w_c = 0.25) gave the
largest criterion improvement (+0.0150 mean), but because the criterion manipulates masks,
the area term changes directly while the appearance term changes only indirectly — so the
criterion **structurally underrates the appearance term**. A criterion cannot select on
something it cannot measure. Of 11 candidates, 2 were adopted and 9 rejected; the record is
in `results/diagnostics/PREREG_branch_improvement.md`.

> **Easy to misread**: `code/selection_criteria/soft_areas.py` iterates over the test split.
> That is *scoring*, not selection — that experiment (soft masks) was rejected on the
> validation tail (pushpins 1.77 → 10.03). Test AUROC is used only for the single final report.

---

## 5. What is not included, and why

| Item | Size | How to regenerate |
|---|--:|---|
| DINOv3-L feature cache | ~18 GB | `code/branches/composition_psad/build_patchcore_pt.py` |
| U-Net segmenter checkpoints | ~9 GB | `code/branches/composition_psad/train_unet_seeds.sh` |
| EfficientAD training weights | ~2 GB | `code/branches/reconstruction_ead/train_ead_seeds.sh` |
| MVTec LOCO AD dataset | ~6 GB | from MVTec Software GmbH |
| Rejected exploratory PSAD variants | — | flags in `score_psad.py` |

**Per-image scores are included**, because they are sufficient for the verification in §1.
The bulk items above are needed only to rebuild those scores from scratch — and, per §1,
rebuilding does not reproduce them exactly.

Not available: the human-annotation variant (5 categories × 3 seeds were never produced, so
the corresponding column was removed from the paper).

---

## 6. Paths

Scripts are shipped **exactly as they ran**, so the repository root is hardcoded in most of
them. Execution records were not edited after the fact. To retarget:

```bash
grep -rl /workspace/ai-vision-research . | xargs sed -i "s|/workspace/ai-vision-research|$PWD|g"
```

`code/analysis/reproduce_from_package.py` contains no absolute path and runs anywhere.
Details in `code/RETARGET.md`.

---

## 7. Integrity

`MANIFEST.md` lists every file with its origin path and SHA-256 prefix. `code/audit/` contains
the scripts that built and checked this package, including a coverage audit that extracts every
numeric value from the manuscript and confirms it is backed by an artifact here.

## License

MIT for our code and derived artifacts — see `LICENSE`. Third-party components (dataset, model
weights, upstream implementations) remain under their own licenses and are not redistributed.
