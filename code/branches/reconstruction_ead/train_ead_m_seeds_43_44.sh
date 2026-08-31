#!/bin/bash
# Train EAD-M (medium) seeds 43, 44 — fill in missing multi-seed for PUAD-M.
# seed42 already in reports/phase0/efficient_ad_official_medium/{cat}_seed42_v1/
set -e
cd /workspace/ai-vision-research

LOG=/tmp/ead_m_seeds_43_44_train.log
> $LOG
echo "Starting EAD-M multi-seed training at $(date)" | tee -a $LOG

for SEED in 43 44; do
    for CAT in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
        OUT=reports/phase0/efficient_ad_official_medium_seeds/${CAT}_seed${SEED}
        if [ -f "$OUT/trainings/mvtec_loco/${CAT}/teacher_final.pth" ]; then
            echo "[skip] EAD-M $CAT seed=$SEED already trained" | tee -a $LOG
            continue
        fi
        echo "==== EAD-M seed=$SEED $CAT $(date +%H:%M:%S) ====" | tee -a $LOG
        mkdir -p "$OUT"
        python3.12 scripts/phase0/run_eads_seed.py $SEED \
          --dataset mvtec_loco \
          --subdataset $CAT \
          --train_steps 70000 \
          --model_size medium \
          --output_dir "/workspace/ai-vision-research/$OUT" \
          --mvtec_loco_path /workspace/ai-vision-research/datasets/MVTecLOCO \
          --weights /workspace/ai-vision-research/external/efficient_ad_official/models/teacher_medium.pth \
          --imagenet_train_path /workspace/ai-vision-research/datasets/imagenet1k/train 2>&1 | tail -10 | tee -a $LOG
        echo "==== EAD-M seed=$SEED $CAT done $(date +%H:%M:%S) ====" | tee -a $LOG
    done
done
echo "ALL DONE $(date)" | tee -a $LOG
