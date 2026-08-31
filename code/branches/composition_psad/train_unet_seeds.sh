#!/bin/bash
# PSAD 재구축 3단계 — CNNSegmenter 학습 + 전 split 분할맵 생성 (시드×범주 순차).
# 공식 run_unet_v2.sh 설정 그대로: 300ep, lr 1e-3, pretrained False(scratch).
# 사용: bash train_unet_seeds.sh 42 [0 1234 ...]
#
# SEED_TAG (선택) — 출력 디렉터리 이름만 바꾸고 **시드 값은 그대로** 쓴다.
#   복제 실행(같은 시드·같은 기계로 다시 학습)에 쓴다. 정본 산출물을 덮지 않기 위함이다:
#     SEED_TAG=42rep2 bash train_unet_seeds.sh 42
#   -> --random_seed 42 로 학습하되 output/unet_seed42rep2/ 에 쓴다.
set -u
REPO=/workspace/ai-vision-research
DATA=$REPO/external/PSAD_official/LOCO_MVTec_AD
LOG=$REPO/_logs/psad_unet_train.log
mkdir -p "$REPO/_logs"

SEEDS="$@"
[ -z "$SEEDS" ] && { echo "사용: $0 <seed> [seed...]"; exit 2; }
echo "=== PSAD unet 학습 시작 $(date) | seeds: $SEEDS ===" | tee -a "$LOG"

for SEED in $SEEDS; do
  TAG="${SEED_TAG:-$SEED}"          # 디렉터리 이름. 시드 값($SEED)과 분리한다.
  for CAT in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
    CKPT=$DATA/output/unet_seed${TAG}/${CAT}/${CAT}_300.pth
    SEGOUT=$DATA/unet_seed${TAG}/${CAT}
    if [ -f "$CKPT" ] && [ -d "$SEGOUT" ]; then
      echo "[skip] seed=$SEED tag=$TAG $CAT 완료됨" | tee -a "$LOG"
      continue
    fi
    echo "==== seed=$SEED tag=$TAG $CAT 시작 $(date +%H:%M:%S) ====" | tee -a "$LOG"
    START=$(date +%s)
    python3.12 "$REPO/scripts/psad_rebuild/train_unet_seed.py" "$SEED" \
      --data_dir "$DATA" \
      --seg_dir csad_pseudo_seg \
      --obj_name "$CAT" \
      --num_epochs 300 \
      --learning_rate 1e-3 \
      --pretrained False \
      --save_dir "unet_seed${TAG}" \
      --snapshot_dir "$DATA/output/unet_seed${TAG}" 2>&1 | tail -3 | tee -a "$LOG"
    END=$(date +%s)
    echo "==== seed=$SEED tag=$TAG $CAT 완료 $(( (END-START)/60 ))분 ====" | tee -a "$LOG"
  done
done
echo "=== 전체 완료 $(date) ===" | tee -a "$LOG"
