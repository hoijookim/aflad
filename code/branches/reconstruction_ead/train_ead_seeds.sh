#!/bin/bash
# EAD-S 임의 시드 학습 (train_ead_seeds_43_44.sh 를 시드 인자화한 일반형).
# 사용: bash scripts/phase0/train_ead_seeds.sh 42 43 44 45 46
# 산출: reports/phase0/efficient_ad_official_small_seeds_std/{cat}_seed{S}/trainings/mvtec_loco/{cat}/*.pth
# 이미 학습된 조합은 건너뛴다(재개 가능).
set -u
cd /workspace/ai-vision-research

SEEDS="$@"
[ -z "$SEEDS" ] && { echo "사용: $0 <seed> [seed...]"; exit 2; }

LOG=_logs/ead_train_seeds.log
mkdir -p _logs
echo "=== EAD-S 학습 시작 $(date) | seeds: $SEEDS ===" | tee -a $LOG

for SEED in $SEEDS; do
    for CAT in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
        OUT=reports/phase0/efficient_ad_official_small_seeds_std/${CAT}_seed${SEED}
        if [ -f "$OUT/trainings/mvtec_loco/${CAT}/teacher_final.pth" ]; then
            echo "[skip] $CAT seed=$SEED 이미 학습됨" | tee -a $LOG
            continue
        fi
        echo "==== seed=$SEED $CAT 시작 $(date +%H:%M:%S) ====" | tee -a $LOG
        rm -rf "$OUT"          # 중단된 잔여물 제거 (efficientad.py 는 기존 디렉토리에 makedirs 실패)
        mkdir -p "$OUT"
        START=$(date +%s)
        python3 scripts/phase0/run_eads_seed.py $SEED \
          --dataset mvtec_loco \
          --subdataset $CAT \
          --train_steps 70000 \
          --model_size small \
          --output_dir "/workspace/ai-vision-research/$OUT" \
          --mvtec_loco_path /workspace/ai-vision-research/datasets/MVTecLOCO \
          --weights /workspace/ai-vision-research/external/efficient_ad_official/models/teacher_small.pth \
          --imagenet_train_path /workspace/ai-vision-research/datasets/imagenet1k/train 2>&1 | tail -3 | tee -a $LOG
        END=$(date +%s)
        echo "==== seed=$SEED $CAT 완료 $(( (END-START)/60 ))분 ====" | tee -a $LOG
    done
done
echo "=== 전체 완료 $(date) ===" | tee -a $LOG
