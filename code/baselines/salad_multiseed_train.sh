#!/bin/bash
# SALAD multi-seed train (Phase A audit completeness)
# 5 cats × 3 seeds (42/43/44) = 15 trainings, ~22h total

set -e
cd /workspace/ai-vision-research/external/SALAD
export SALAD_NUMPY_PATCH=1

LOCO_PATH="/workspace/ai-vision-research/datasets/MVTecLOCO"
RESULTS_DIR="/workspace/ai-vision-research/reports/phase0/salad_reproduction_multiseed"
mkdir -p "$RESULTS_DIR"

echo "=== SALAD Multi-seed train (3-seed) ==="

for seed in 42 43 44; do
    for cat in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
        OUT_DIR="$RESULTS_DIR/seed${seed}/${cat}"
        if [ -f "$OUT_DIR/scores.json" ]; then
            echo "[skip] seed=$seed cat=$cat already done"
            continue
        fi
        mkdir -p "$OUT_DIR"
        echo ""
        echo "=== seed=$seed $cat $(date +%H:%M:%S) ==="
        python3.12 train_salad.py \
            --category "$cat" \
            --mvtec_loco_path "$LOCO_PATH" \
            --output_dir "$OUT_DIR" \
            --weights "models/teacher_medium.pth" \
            --imagenet_train_path "/workspace/ai-vision-research/datasets/imagenet1k/train" \
            --seed "$seed" \
            2>&1 | tee "$OUT_DIR/train.log" | tail -30
    done
done
echo "ALL DONE $(date)"
