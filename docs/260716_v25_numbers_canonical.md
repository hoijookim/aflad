# [v25 수치 정본] AFLAD 논문 전 수치 단일 참조 — 표·산문·출처·변경이력

> ## ⚠ 이 문서는 supersede 되었다 (2026-08-19)
>
> 논문이 **test-free 를 주 설정으로 전환**하면서 아래 수치들이 바뀌었다. 현행 정본은
> [`260817_testfree_hc_canonical.md`](260817_testfree_hc_canonical.md) 와
> [`260817_baseline_seed_audit.md`](260817_baseline_seed_audit.md) 다.
>
> | 이 문서 | 현행 | 사유 |
> |---|---|---|
> | 헤드라인 0.9815 ± 0.0029 (hc, 5시드) | **0.9723 ± 0.0011** (test-free hc, 3시드 {42,43,44}) | 구 헤드라인은 정규화·집계를 test AUROC 로 골랐다 — test-free 가 아니다 |
> | test-free 변형 0.9751 ± 0.0035 (시드 {0,42,1234}) | 동일 프로토콜 재구축 0.9733 ± 0.0007 | 시드 통일. 차이는 구 표준편차의 0.51배로 통계적으로 구분되지 않는다 |
> | EAD-M 0.7846 | **0.8971 ± 0.0047** | 구 값은 penalty 정칙화가 imagenette 폴백이었다 (출판 순서가 뒤집혀 있었다) |
> | DINOv3-L PatchCore 0.7879 | **0.7890 ± 0.0018** | 메모리 뱅크에서 validation 제외 (train-only 통일) |
>
> **그대로 유효한 것**: SALAD 재현 0.9469, FPS, AU-sPRO, 데이터 분할·프로토콜 서술.
> 이 문서를 남기는 이유는 무엇이 왜 바뀌었는지가 기록이기 때문이다.


**작성일**: 2026-07-16 · **기준**: `260716_AFLAD_v25_word_source.md` (= `AFLAD_paper_v25.docx`, 219문단/표14/그림11)
**용도**: v25에 들어가는 **모든 수치**를 표 번호 그대로 한 곳에 모은 단일 참조. 수치 갱신 시 이 문서와 v25 소스를 함께 갱신할 것.
**주의**: 구 수치 문서(`260624_full_results_baselines_metrics.md`=hcp 0.9796 세대, `260630_all_tables_6to15.md`=구 표번호 세대)는 이 문서가 **supersede**.

---

## 0. 헤드라인 (초록·기여·결론)

| 항목 | 값 | 비고 |
|---|---|---|
| **image-level AUROC (L+S)** | **0.9815 ± 0.0029** | annotation-free(주 설정), 5-seed {42–46}, hc |
| vs 동일환경 SALAD 재현 | **+3.5%p** (0.9469 대비) | Welch t≈7.7, df≈2.4, 양측 p≈0.010 |
| 처리량 | **~34 FPS** (~29 ms/img) | RTX 5090, batch 1, FP32, 순차 합산 |
| 완전 test-free 배포 변형 | **0.9751 ± 0.0035** | 3-seed {0,42,1234}, val-good z-norm + 등가중(1,1,1); SALAD 대비 +2.8%p |
| AUPR / 최대 F1 (L+S) | **0.9825 ± 0.0032 / 0.9602 ± 0.0057** | 동일 fused score (§5.4) |
| 국소화 AU-sPRO@0.05 | **0.5291** (val 0.5322) | held-out 반분 (§5.5) |

---

## 1. 표 1 — 세 분기 설정 (§3.1, 수치 항목만)

| 항목 | 재구성 (EfficientAD-S) | 패치 메모리 | 구성 (annotation-free) |
|---|---|---|---|
| backbone | WRN→PDN (수용영역 33×33) | DINOv3-L/16 frozen, **L17** | CNNSegmenter(WRN-101-2+좌표채널) · 임베딩 ResNet-101 |
| 입력 | 256² | 336² → 21×21×1024 | 256² (의사레이블 지도 512²) |
| 학습 | student **70,000 steps** | 없음 (뱅크만) | **300 ep** · 10·CE+Dice · AdamW **lr 1e-3** · batch **2** · scratch |
| 의사레이블 K | — | — | **7 / 9 / 26(5×5+bg) / 7 / 10** (bfast/juice/pushpins/screw/splicing) |
| 메모리 | — | 정상 패치 **50,000** 무작위 (coreset 아님) | 면적 히스토그램+구성 임베딩 뱅크 |
| 추론 지연 | 3.45 ms | ≈13.4 ms (최대 항목) | ≈11.2 ms |

## 2. 표 2 — 융합·국소화 설정 (§3.2–3.3, 수치 항목만)

- 융합 CV: 외부 stratified **5-fold × 3 repeats** / 내부 **4-fold** · 가중 격자 **{0,.2,.4,.6,1}³ ∖ {0}** (축별)
- 보고 AUROC = 15개 외부 폴드 평균 (pooled OOF와 **±0.001** 이내 일치)
- 국소화: z(Maha L17)+z(패치메모리 지도)+z(DINOv2-B L11) **등가중 합** · 해상도 **512 (32×32)** · Maha 공분산 수축 1e-3 · 국소화 전용 뱅크 40,000 · 테스트 val/test 반분

## 3. 표 3 — 직교성×신호 통제 검정 (§4.2)

| 신호원 | 신호(자체) | 직교성(상관) | fusion-delta |
|---|---|---|---|
| D-SEG (면적+색) | 강 0.953 | 낮음 0.86 | −0.07pp |
| D-B (VLM 관계) | 약 0.582 | 높음 0.055 | −0.07pp |
| 구성 (PSAD) | 강 0.989 | 직교 −0.01 | **+7.6pp** |

†측정 범주: D-SEG=breakfast_box(신호는 5-cat 평균) / D-B=splicing / PSAD=screw_bag.
**VisA 검증(§4.2)**: 18축×4범주(n=72) · 예측자=(AUROC−0.5)×(1−평균|상관|) · **상위6 > 하위6 융합, 4/4 범주** · pooled Spearman 0.767(기술치 — 음성통제 ρ≈0.84로 **헤드라인 아님**) · 결합−신호단독 증분 **+0.043**(0.767 vs 0.724; 노이즈 통제에서 소멸) · Fisher-z ρ̄=0.900 (95% CI [0.511, 0.983], 클러스터 4).

## 4. 표 4 — 융합 방법 비교 (§4.3, 5-seed hc)

| 융합 방법 | L+S | Δ vs designed |
|---|---|---|
| **designed (z-norm+축별 가중)** | **0.9815 ± 0.0029** | — |
| uniform-z | 0.9756 ± 0.0031 | −0.59pp |
| RandomForest 메타 | 0.9756 ± 0.0032 | −0.59pp |
| best-single (oracle 참고) | 0.9709 ± 0.0028 | −1.06pp |
| raw-mean (z-norm 無) | 0.7974 ± 0.0055 | **−18.41pp** |

산문 델타: raw→uniform-z **+17.8%p** · 축별−균일 +0.59 · designed−RF +0.59 · designed−best-single +1.06. designed−uniform은 5/5 시드 양수(paired).

## 5. 표 5 — 동일환경 image-level AUROC(L+S) (§5.2)

| Method | venue | L+S | logical | structural | n_seed | 출판값(참고) |
|---|---|---|---|---|---|---|
| **AFLAD (annotation-free, 주)** | — | **0.9815 ± 0.0029** | 0.9796 ± 0.0038 | 0.9833 ± 0.0020 | 5 | — |
| AFLAD (human-annotation, 참고) | — | 0.9752 ± 0.0048 | 0.9715 ± 0.0094 | 0.9789 ± 0.0014 | 5 | — |
| SALAD | ICCV'25 | 0.9469 ± 0.0074 | 0.9497 | 0.9441 | 3 | 0.961 |
| PUAD-M / -S | ICIP'24 | 0.9284 / 0.9276 | 0.9148 / 0.9100 | 0.9420 / 0.9453 | 3 / 1 | 0.931 |
| EAD-S | WACV'24 | 0.8966 ± 0.0027 | 0.8554 | 0.9378 | 5 | 0.900 |
| ComAD | AEI'23 | 0.7987 | 0.8691 | 0.7282 | 3 | 0.898 |
| DINOv3-L PatchCore | — | 0.7901 ± 0.0056 | 0.6665 | 0.9137 | 5 | — |
| DINOv3-B / DINOv2-B PatchCore | — | 0.7999 / 0.7570 | 0.729 / 0.665 | 0.870 / 0.849 | 1 | — |
| EAD-M | WACV'24 | 0.7846 | 0.7837 | 0.7854 | 3 | 0.907 |

각주: PUAD-M·ComAD·EAD-M pooled 시드 std 0.004–0.007. 산문 참고 출판값: EAD-M 90.7 / EAD-S 90.0 / PUAD 93.1. SALAD 재현−출판 갭 −1.4%p.

## 6. 표 6 — 범주별 단일 분기·융합 (§5.3, 5-seed)

| 범주 | EAD | PC | 구성 | 융합(anno-free) | human-anno(참고) |
|---|---|---|---|---|---|
| breakfast_box | 0.8335 | 0.8672 | 0.8535 | 0.9653 ± 0.0052 | 0.9676 ± 0.0162 |
| juice_bottle | 0.9898 | 0.8637 | 0.8966 | 0.9928 ± 0.0048 | 0.9968 ± 0.0031 |
| pushpins | 0.9724 | 0.7053 | 0.9278 | 0.9884 ± 0.0053 | 0.9748 ± 0.0113 |
| screw_bag | 0.7317 | 0.6921 | **0.9485** | 0.9885 ± 0.0026 | 0.9718 ± 0.0124 |
| splicing_connectors | 0.9555 | 0.8221 | 0.8265 | 0.9722 ± 0.0026 | 0.9651 ± 0.0071 |
| **평균** | **0.8966** | **0.7901** | **0.8906** | **0.9815 ± 0.0029** | 0.9752 ± 0.0048 |

산문: 쌍별 융합 EAD+PC **0.9165** / PC+구성 **0.9545** / EAD+구성 **0.9669** < 3소스 0.9815. anno-free가 5시드 중 4개 우위(평균 +0.63pp). 그림10: 구성 기여 screw_bag **+23.96±0.29pp**, 5-cat 평균 **+6.49±0.17pp** (EAD+PC 대비).

## 7. §5.4 다중 지표 (동일 fused score)

AUROC **0.9815±0.0029** · AUPR **0.9825±0.0032** · 최대 F1 **0.9602±0.0057** (5-seed {42–46}).

## 8. 표 7 — 국소화 (§5.5, held-out)

| 지표 | AFLAD | Hybrid baseline | Δ |
|---|---|---|---|
| AU-sPRO@0.05 | **0.5291** (val 0.5322) | 0.394 | +13.5pp |
| pixel-AUROC (mean) | 0.755 | 0.646 | +10.9pp |
| pixel-AUROC (structural) | 0.911 | 0.815 | +9.6pp |
| pixel-AUROC (logical) | 0.729 | 0.632 | +9.7pp |

Hybrid baseline = DINOv2-B 지도 단독(자기 절제). 단일 시드·고정 반분(rs=42).

## 9. 표 8 — 추론 지연 (§5.6, RTX 5090 batch 1)

| Method | ms | FPS |
|---|---|---|
| **AFLAD** | **29** | **34** |
| PUAD-S | 2.26 | 442 |
| PaDiM | 2.38 | 419 |
| EfficientAD-S | 3.45 | **290** |
| EfficientAD-M | 3.9 | 230 |
| PUAD-M | 3.95 | 225 |
| CSAD (feat-only 추정) | 5.7 | 175 |
| PatchCore-WRN50 | 6.36 | 157 |
| DINOv3-L PatchCore | 13.4 | 70 |
| SALAD | 57 | 17 |
| ComAD | 57 | 17.5 |

산문(§5.6): 분기 합 28.1 ms + 융합 오버헤드 = 29 ms · DINOv3-L 분기 ≈46% · 분산 배치 이론 하한 ≈13.4 ms(미실측) · DINOv3-L 재측정 중앙값 14.8 ms(24층 전체 상한)와 부합 · **파라미터 합 ≈4.8억**(8.1M+303M+~171M) · **뱅크 ≈200 MB**(fp32) · 지배 분기 **peak VRAM ≈1.7 GB**.

## 10. §5.7 재현성 산문 수치

seed 45 재계산 = 저장값 정확 일치 · 3→5시드 std 0.0020→0.0029 · uniform-z 0.9756(유형 무구분으로도 성립).

## 11. 부록 A — 의사레이블 설정 (표 A1 수치 항목)

box/text threshold: bfast 0.25/0.20 · juice 0.30/0.30 · pushpins 0.15/0.20 · screw 0.30/0.30 · splicing 0.30/0.20. K = 7/9/26/7/10.
재구축: MeanShift **bandwidth 3.5**(위치 가중 2) · ResNet-101 **layer1–4** 마스크 평균 · s_comp 정규화 = train 정상 leave-one-out 1-NN 최댓값 · 의사레이블 512²→256² 최근접 보간.

## 12. 부록 B — 시드별 원값

**B1 주 설정(anno-free)**: 42=0.9818(L 0.9796/S 0.9840) · 43=0.9802(0.9781/0.9822) · 44=0.9802(0.9775/0.9829) · 45=0.9863(0.9861/0.9864) · 46=0.9788(0.9766/0.9811)
**B2 human-anno**: 42=0.9782 · 43=0.9761 · 44=0.9808 · 45=0.9729 · 46=0.9683
**B3 test-free**: 0=0.9718 · 42=0.9788 · 1234=0.9748 → **0.9751±0.0035**

## 13. 부록 C — 이중 분리 (5회 추첨 평균±std)

**표 C1 (readout 고정, backbone 스케일)**

| 범주 | backbone | params | structural | cardinality | layer |
|---|---|---|---|---|---|
| screw_bag | DINOv3-S/16 | 21M | 0.860±0.014 | 0.580±0.013 | L3 |
| screw_bag | DINOv3-B/16 | 86M | 0.884±0.014 | 0.579±0.029 | L3 |
| screw_bag | DINOv3-L/16 | 300M | 0.907±0.009 | 0.562±0.020 | L8 |
| screw_bag | DINOv2-B/14 | 86M | 0.870±0.011 | 0.468±0.014 | L11 |
| screw_bag | DINOv2-L/14 | 300M | 0.888±0.013 | 0.508±0.032 | L17 |
| pushpins | DINOv3-S/16 | 21M | 0.787±0.008 | 0.594±0.013 | L11 |
| pushpins | DINOv3-B/16 | 86M | 0.822±0.019 | 0.538±0.018 | L6 |
| pushpins | DINOv3-L/16 | 300M | 0.844±0.013 | 0.513±0.015 | L17 |

산문(§4.1): 구조 screw 0.860→0.907 / pushpins 0.787→0.844 단조(DINOv3 계열 내) · 개수 8조합 전부 **0.47–0.59** · 최대모델(0.562) < 최소모델(0.580).

**표 C2 (표현 고정 L17, readout 교체)**

| readout | screw 개수 | screw 구조 | pushpins 개수 | pushpins 구조 |
|---|---|---|---|---|
| patch-memory NN | 0.485±0.011 | 0.899±0.006 | 0.513±0.015 | 0.844±0.013 |
| area-histogram | **0.725±0.015** | 0.747±0.008 | 0.633±0.013 | 0.552±0.023 |
| instance-count | 0.634±0.027 | 0.622±0.010 | **0.743±0.011** | 0.509±0.027 |
| set-distance | 0.638±0.004 | 0.750±0.007 | 0.549±0.006 | 0.696±0.004 |

복원: screw 0.485→**0.725**(면적) · pushpins 0.513→**0.743**(계수).
**하위유형 전수(부록 C 말미)**: 논리 45+구조 5 · 논리 AUROC 범위 **0.38–1.00** · empty_bottle 1.00 · extra_cable 0.90 · screw 계수형 8건 중 7건 0.38–0.56(예외 missing_long_screw 0.73) · 수량·크기형 4건 0.44–0.60 · breakfast 과일비율 0.80 · **VisA PCB 누락 pcb1 0.904 / pcb3 0.926**.

## 14. 본문 산문에만 있는 기타 수치

§1: 육안검사 결함 누락 20–30%[2] · 숙련 적중률 85%[3] · 식스시그마 3.4 DPMO[4] · MVTec AD 이상 1258 중 논리 37(~97% 구조).
§2: MVTec AD 2 최고 58.7%(AU-PRO₀.₃₀)[15] · GCAD 재현 ~83.3%[8] · ComAD 논리 90.1/평균 89.8[11] · PSAD 논리 98.1/평균 94.9[9] · CSAD 평균 95.3(구조 94.0/논리 96.7)[12] · SALAD 96.1[13] · PUAD 93.1[29] · SINBAD 88.3[30] · SLSG 90.3[31].

---

## 15. 출처 데이터 매핑 (수치 → JSON)

| v25 위치 | 소스 |
|---|---|
| 표 4 / B1 per-seed | `reports/countgd/naive_fusion_hc.json` (designed=헤드라인 bit-일치) |
| 표 5 AFLAD 축별 | `reports/countgd/v4free_peraxis_table10.json` |
| 표 5 baseline | `reports/path_y/metric_unify/ls_canonical_scores.json` + `reports/countgd/baseline_5seed_peraxis.json` (EAD-S/DINOv3-L 5-seed 통일) |
| 표 6 per-cat / 그림 10 | `reports/countgd/percat_table_fill.json`(+`_anno`) · `fusion_hc_percat_perseed.json` |
| 쌍별 융합·구성 단독 | `reports/countgd/hc_pairwise_standalone.json` (무결 검수 `audit_hc_pairwise.py` PASS) |
| §5.4 다중지표 | `reports/countgd/v4free_metrics_5seed_hc.json` |
| 표 7 국소화 | `reports/salad_free/p18_heldout_fullres.json` · `p21_pixel_auroc.json` (Hybrid=`p4_hybrid_baseline.json`) |
| 표 8 지연 | `reports/path_y/fps_comparison_260621.json` (EAD-S 290=branches 실측; baselines 368은 출판값 혼입으로 폐기) |
| 표 3 / VisA | `reports/countgd/{dsq_color_composition_full_table,db_vlm_result}.json` · `d5_visa_principle.json` |
| 표 C1·C2 | `reports/countgd/d2d3_multidraw_tables67.json` (5-draw 재측정) |
| B2 / B3 | `psad_fusion_v4free_5seedconsec_anno.json` · `val_znorm_fusion.json` |
| 하위유형 전수 / PCB | `reports/countgd/footprint_resolved_allcat.json` · `d2d3_crossdataset_pcb{1,3}.json` |
| §5.6 메모리·VRAM | v25 신규 실측 (commit 0382a9a) |
| 서지 검증 | `docs/260716_refs_audit.json` (50편) |

## 16. 구버전 대비 변경 이력 (수치 감사 추적)

| 수치 | 구 (문서/세대) | v25 | 사유 |
|---|---|---|---|
| 헤드라인 L+S | 0.9796±0.0023 (hcp) | **0.9815±0.0029** (hc) | B 채택(구성 'p' 제거, 260630) — "설계 정리"이지 개선 주장 아님, juice −0.48pp trade-off |
| AUPR / F1 | 0.9801 / 0.9566 | 0.9825 / 0.9602 | hc 재측정 |
| human-anno | 0.9753 | 0.9752 | v19 정정 |
| SALAD std | ±0.0060 | **±0.0074** | ddof=1 (v19) |
| EAD-S / DINOv3-L PC | 0.8978(3s) / 0.7913(1s) | 0.8966(5s) / 0.7901(5s) | seed-set 5-seed 통일 |
| D2 구조 L (screw) | 0.9056 (단일 추첨) | **0.907±0.009** (5-draw) | rng 우연고점 정정(v16 M1); D3 0.911도 동일 사유 폐기 |
| D3 복원 (screw) | 0.504→0.732 (단일) | **0.485→0.725** (5-draw) | 〃 |
| EAD-S FPS | 368 | **290** | 출판값 혼입 → branches 실측 |
| 국소화 해상도 | 672 (42×42) | **512 (32×32)** | 코드검증(docstring 잔재) |
| VisA 0.767 | 헤드라인 | **기술치로 격하** | 음성통제 ρ≈0.84; 근거=상위/하위 분리+증분 +0.043 (v24) |
| 그림 p=4e-15 | 표기 | **제거** | 독립성 위반 무효 통계량 |
| juice K | 8 | **9** | 코드검증 |
| 파라미터/뱅크/VRAM | 미보고 | 4.8억 / 200MB / 1.7GB | v25 신규 실측 |

---
*정합 검증: 표 4 designed·§5.4 AUROC·B1 평균 = 0.9815 상호 일치 · 표 6 평균 = 표 5 AFLAD 행 일치 · C1/C2 = 그림 1 캡션 "부록 C와 동일" 선언과 일치 · §4.1 산문(0.860→0.907, 0.485→0.725, 0.47–0.59) = C1/C2 표값 일치.*
