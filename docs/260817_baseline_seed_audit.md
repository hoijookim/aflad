# 표 5 베이스라인 감사 — 시드 정렬 + 결함 1건

작성: 2026-08-17 (GPU 세션)

## 1. 시드 정렬: 전 방법 {42, 43, 44} 완료

추가 학습은 필요 없었다. 시드별 원값이 모두 남아 있어 재집계·재채점으로 끝났다.

| 방법 | 구 표기 | **{42,43,44}** | 산출물 |
|---|--:|--:|---|
| AFLAD hc (구 주 설정, CV) | 0.9815 ± 0.0029 (5s) | **0.9807 ± 0.0009** | `realign_3seed_42_43_44.json` |
| AFLAD hcp (절제, CV) | 0.9796 ± 0.0023 (5s) | 0.9795 ± 0.0006 | 동일 |
| **AFLAD test-free hc (신 주 설정)** | — | **0.9723 ± 0.0011** | `testfree_final_eadfix_3seed.json` |
| SALAD | 0.9469 ± 0.0074 (3s) | **0.9469 ± 0.0074** (동일) | `realign_salad_3seed.json` |
| PUAD-M | 0.9284 (3s) | **0.9284 ± 0.0042** | `realign_puad_3seed.json` |
| EAD-S | 0.8966 ± 0.0027 (5s) | **0.8945 ± 0.0041** | `realign_eads_pc_3seed.json` |
| ComAD | 0.7987 (3s) | **0.7987** (동일, std 산출 불가) | `ls_canonical_scores.json` |
| DINOv3-L PatchCore | 0.7901 ± 0.0056 (5s) | **0.7890 ± 0.0018** | `realign_eads_pc_3seed.json` |
| EAD-M | 0.7846 (3s) | **0.7846 ± 0.0106** → **0.8971 ± 0.0047 로 대체 권고 (§3)** | `realign_eadm_fullimagenet_3seed.json` |

통합: `reports/countgd/table5_3seed_42_43_44.json`
test-free 주 설정은 SALAD 재현 대비 **+2.54pp**.

## 2. 시드가 무엇을 바꾸는가 — 전 방법이 자체 학습이다

"저자 공개 가중치를 쓰니 시드 변동이 없다"는 우려를 확인했고, **사실이 아니다.**

| 방법 | 학습 주체 | 시드가 바꾸는 것 |
|---|---|---|
| AFLAD | 우리 | 전체 학습 (EAD·PSAD UNet·PC coreset) |
| SALAD | **우리** (`path_X/salad_multiseed_train.sh`, 3시드 × 5범주 × 70k iter) | 전체 학습 |
| PUAD-M | **우리** (시드별 EAD-M 학습 + 공식 PUAD Mahalanobis 적합) | 전체 학습 |
| EAD-S / EAD-M | 우리 (공식 EfficientAD 코드) | 전체 학습. teacher 만 ImageNet 사전학습 — EfficientAD 규약 |
| DINOv3-L PC | 우리 | coreset 샘플링. 백본은 사전학습 DINOv3 — 설계상 그러함 |
| ComAD | 우리 | 전체 |

`reports/phase0/salad_reproduction`(저자 가중치 1회 추론, auc_combined 95.13)과
`reports/path_y/puad_leapmind`(LeapMind 가중치 검증)는 **초기 검증용 산출물**이고
표 5 의 원천이 아니다. 표 5 는 `path_y` 의 시드별 자체 학습 결과를 쓴다.

### SALAD 다중시드의 철회와 수정 (경위 기록)
`docs/path_y_salad_multiseed_RETRACTED_260603.md` 가 다중시드 AUROC 0.747 을 철회했다 —
SALAD 의 `predict()` 가 요구하는 런타임 통계 7종(teacher_mean/std, 각종 quantile,
Mahalanobis 분포)이 체크포인트에 저장되지 않아 screw_bag 이 0.500(무작위)으로 나왔다.
그 다음날(2026-06-04) `PY_salad_multiseed_v2_FULL.py` 로 재산출한 `salad_multiseed_v2` 가
현재 표 5 의 원천이고, screw_bag 이 0.9396 / 0.9517 / 0.9263 으로 정상이다. 철회 사유는
해소됐다.

### SALAD 수치 셋 (혼동 방지)
| 값 | 정체 |
|--:|---|
| 96.1% | SALAD **원논문 발표값** (pooled image AUROC) |
| 95.13% | 저자 공개 가중치 1회 추론 재현 `auc_combined` 5범주 평균 (−0.97pp) |
| 0.9469 | 표 5 의 **L+S** = 0.5×(logical+structural), 우리 3시드 학습 |

95.13 → 96.1 격차는 breakfast_box 하나에서 온다(원논문 full ImageNet teacher 대신
Imagenette 대체 → 그 범주만 84.0%, −10pp). 앞 두 값은 pooled AUROC 이고 0.9469 는
논리·구조를 따로 재 평균한 것이므로 **직접 비교 불가**.

## 3. 결함 1건 — 표 5 의 EAD-M 이 imagenette 폴백으로 학습돼 과소평가됐다

EAD-M 학습이 **두 갈래로 존재하고 11pp 차이**가 난다. 둘 다 우리 학습이다.

| 경로 | 구현 | penalty 정칙화 데이터 | L+S | 용도 |
|---|---|---|--:|---|
| `reports/phase0/efficient_ad_medium` | anomalib | **imagenette 폴백** | **0.7846** | **표 5 의 EAD-M** |
| `reports/phase0/efficient_ad_official_medium*` | 공식 EfficientAD | **full ImageNet** | **0.8971** | PUAD-M 의 기반 |

EfficientAD 의 penalty 항은 ImageNet 이미지로 student 의 과잉 일반화를 막는다.
imagenette(10클래스 소규모)로 대체하면 그 효과가 크게 약해진다 — SALAD breakfast_box 에서
−10pp 를 낸 것과 **같은 이탈**이다. `run_ead_m_loco.py` 주석도 "Penalty regularizer dir =
imagenette (anomalib default; full ImageNet not …)" 로 이탈을 명시하고 있다.

**두 가지 문제가 따라온다.**

1. **EAD-S 와 조건이 다르다.** EAD-S 는 `train_ead_seeds.sh` 에서
   `--imagenet_train_path .../imagenet1k/train` 으로 **full ImageNet** 을 썼다.
   같은 표에서 한 방법만 열등한 정칙화 데이터로 학습된 상태다.

2. **출판 순서가 뒤집혔다.** 출판값은 EAD-M(90.7) > EAD-S(90.0) 인데 표 5 는
   EAD-M(0.7846) << EAD-S(0.8945) 다. full ImageNet 으로 학습한 EAD-M(0.8971)은
   EAD-S(0.8945)를 상회해 **출판 순서와 일치**한다. 즉 현재 표 5 의 순서 역전은
   방법 차이가 아니라 학습 설정 차이의 산물이다.

### 보완 완료 (2026-08-17)

**대체값 산출 완료: EAD-M (full ImageNet) = 0.8971 ± 0.0047**
(logical 0.8590 / structural 0.9352, 시드별 0.8920 / 0.9013 / 0.8980)
구 표기(imagenette 0.7846 ± 0.0106) 대비 **+11.25pp**.
산출물 `reports/countgd/realign_eadm_fullimagenet_3seed.json`,
스크립트 `scripts/psad_rebuild/realign_eadm_fullimagenet.py`.

**출처 검증**: `official_medium` 체크포인트는 원본 머신에서 소실됐으나 시드별 이미지 점수가
남아 있다. seed42 에서 정본 npz(`efficient_ad_official_medium/npz/`)와 PUAD npz 의 `ead` 열이
**상관 0.9999~1.0000, L+S 소수 3자리 일치** — 같은 모델임을 확인한 뒤 3시드 모두 `ead` 열로
일관되게 산출했다(두 산출물을 섞지 않았다).

**효과**: 교체하면 순서가 출판값과 맞는다 — EAD-M(0.8971) > EAD-S(0.8945), 출판 EAD-M 90.7 >
EAD-S 90.0. 그리고 PUAD-M(0.9284)이 **자기 기반 EAD-M 위에 올라간** 형태가 되어 델타가
+14.4pp 에서 **+3.1pp** 로 정정된다 — 기존 +14pp 는 대부분 서로 다른 EAD-M 학습을 비교한
산물이었다.

교체 시 함께 갱신할 것: 표 5 의 EAD-M 행, §4 의 PUAD-M vs EAD-M 델타 서술,
각주의 "EAD-M pooled 시드 std" 값.

## 4. 산출물

| 내용 | 파일 |
|---|---|
| 통합 표 5 (3시드) | `reports/countgd/table5_3seed_42_43_44.json` |
| SALAD 재집계 | `reports/countgd/realign_salad_3seed.json` |
| EAD-M 재집계 (imagenette 판) | `reports/countgd/realign_eadm_3seed.json` |
| EAD-M (full ImageNet 판) + PUAD L+S | `reports/countgd/realign_puad_3seed.json` |
| EAD-S / DINOv3-L PC | `reports/countgd/realign_eads_pc_3seed.json` |
| AFLAD / ComAD | `reports/countgd/realign_3seed_42_43_44.json` |
| 재집계 스크립트 | `scripts/psad_rebuild/realign_{3seed,eads_pc,puad}.py` |
