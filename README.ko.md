# AFLAD — 재현 패키지

MVTec LOCO AD 논리·구조 이상탐지. 논문에 실린 수치를 **직접 다시 계산해 확인할 수 있게** 만든
패키지다. "우리 말을 믿어라"가 아니라 "돌려 보라"가 목적이다.

**주 설정 — image-level AUROC (L+S) = 0.9723 ± 0.0011** (시드 42/43/44)
logical 0.9685 · structural 0.9762 · 동일환경 SALAD 재현(0.9469) 대비 **+2.54%p**

---

## 1. 30초 만에 검증하기

per-image 점수만으로 헤드라인이 재계산된다. 18GB 특징 캐시도, 9GB 체크포인트도, GPU도
필요 없다 — `numpy` 와 `scikit-learn` 이면 된다.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```

> **Windows**: JSON 산출물에 한국어가 들어 있어 UTF-8 로 돌려야 한다 —
> `set PYTHONUTF8=1`(또는 `python -X utf8 ...`). 스크립트가 `encoding="utf-8"` 를
> 명시하므로 필수는 아니고 안전망이다. `MANIFEST.md` 의 해시는 **LF** 줄바꿈 기준이며
> `.gitattributes` 가 그것을 고정하므로 `core.autocrlf=true` 인 clone 도 일치한다.

```
  seed 42: L+S 0.9721 (logical 0.9662 / structural 0.9780)
  seed 43: L+S 0.9714 (logical 0.9682 / structural 0.9745)
  seed 44: L+S 0.9735 (logical 0.9710 / structural 0.9759)

  3-seed L+S = 0.9723 ± 0.0011
    LS         재현 0.9723  논문 0.9723  차이 0.00001  PASS
    logical    재현 0.9685  논문 0.9685  차이 0.00005  PASS
    structural 재현 0.9762  논문 0.9762  차이 0.00004  PASS
```

### 왜 점수를 배포하는가 — 코드만으로는 부족하다

같은 코드·같은 의사레이블·같은 시드로 **다시 학습하면 분기 점수가 달라진다.** 복제 실행으로
쟀다(같은 기계, seed 42, 독립 재학습):

```
비교 종류                Spearman
시드만 다름               0.9381
기계만 다름               0.9378
실행 시점만 다름           0.9388     <- 아무것도 안 바꿔도 같은 크기로 갈린다
```

원인은 `train_normal_unet.py` 의 `cudnn.benchmark = True` 다 — 합성곱 알고리즘을 실행 시각
측정으로 고르고 상당수가 backward 에서 `atomicAdd` 를 쓴다. 시드는 **초기화**를 고정하지만
(미학습 모듈의 가중치가 두 실행에서 비트 단위로 같다) **연산**을 고정하지 못한다.

**융합 수치는 그래도 재현된다** — 분기 차 −0.0026 이 융합에서 −0.0007 로 줄어 시드
표준편차(0.0011)의 0.61배다. 즉 **"분기 점수가 재현된다" 가 아니라 "융합 수치가 재현된다"**
가 이 방법의 정확한 재현성 주장이고, 그것을 확인하려면 **점수 파일 자체가 있어야 한다.**
근거: `results/tables/psad_variance_decomp.json`.

이 스크립트가 하는 계산이 논문이 선언한 test-free 프로토콜의 전부다: 세 분기 점수를 **공식
검증 분할의 정상 표본** 평균·표준편차로 z-정규화하고, **등가중 (1,1,1) 합**, 결함 유형별
AUROC, `L+S = (logical + structural)/2`. 로그 스케일도 로버스트 통계도 클리핑도 없다.

---

## 2. 방법

세 분기의 점수 합이다. 세 가지가 서로 다른 실패 양식을 잡는다.

| 분기 | 무엇을 보나 | 코드 |
|---|---|---|
| **재구성** EfficientAD-S | 국소 텍스처·구조 이탈 | `code/branches/reconstruction_ead/` |
| **패치 메모리** DINOv3-L L17 PatchCore | 정상 패치 다양체로부터의 거리 | `code/branches/patch_memory_pc/` |
| **구성** PSAD-hc | 부품 면적 히스토그램(h) + 외형 임베딩(c) | `code/branches/composition_psad/` |

구성 분기의 부품 레이블은 CSAD 의사레이블로 학습한 UNet 분할에서 나온다. 여기에 **개선 2건**이
들어간다 — 둘 다 train/val 만 보고 채택했다.

- **혼동쌍 자동 병합** — train 의사레이블과 UNet 예측의 혼동행렬에서 `C[i,j]+C[j,i] > 0.10`
  인 클래스 쌍을 합친다. 5범주 전체에서 screw_bag `c4↔c5` 하나만 걸린다(3시드 0.168/0.261/0.142,
  차순위는 전 범주 ≤ 0.012 — **10배 격차**라 임계값 하나가 범주별 수작업 없이 그 쌍만 집는다).
- **회전 TTA** — 학습 증강 각도와 같은 각도로 추론해 평균. "val 정상 점수 꼬리의 평균과 최댓값을
  둘 다 줄일 것"이라는 규칙을 5범주에 동일 적용해 screw_bag 만 통과했다.

---

## 3. 논문 요소 → 산출물 → 코드

| 논문 요소 | 산출물 | 생성 코드 |
|---|---|---|
| 주 설정 헤드라인 0.9723 | `results/main/testfree_final_eadfix_3seed.json` | `code/fusion/prereg_testfree.py --protocol paper --kind hc --psad-tag _tta_merge` |
| 구 프로토콜 정합 변형 0.9733 | `results/main/testfree_final_eadfix_origproto_3seed.json` | 같은 스크립트 (EAD·PC 시드평균) |
| hcp 변형 (부록 비교) | `results/main/testfree_final_hcp_3seed.json` | 같은 스크립트 `--kind hcp --psad-tag _tta_merge_k5` |
| 기준선 비교표 | `results/tables/table5_3seed_42_43_44.json` | `code/baselines/realign_*.py` |
| PatchCore 계열 (train-only 뱅크) | `results/tables/patchcore_family_trainonly_3seed.json` | `code/branches/patch_memory_pc/patchcore_family_trainonly.py` |
| EAD-M full-ImageNet 교체 | `results/tables/realign_eadm_fullimagenet_3seed.json` | `code/baselines/realign_eadm_fullimagenet.py` |
| 분기 단독·소거 실험 | `results/tables/testfree_supplements_3seed.json` | `code/analysis/testfree_supplements.py` |
| 속도 | `results/tables/fps_comparison_260621.json` | — |
| 국소화 AU-sPRO | `results/tables/p18_heldout_fullres.json`, `p21_pixel_auroc.json` | — |
| EAD-S 5시드 재현 검증 | `results/reproduction/ead_5seed_verdict.json` | `code/audit/ead_5seed_verdict.py` |
| 설계 선택의 사전등록 | `results/diagnostics/PREREG_branch_improvement.md` | — |
| 채택·기각 판정 근거 | `results/diagnostics/criterion_*.json`, `sweep_*.json` | `code/selection_criteria/` |

per-image 점수의 디렉터리별 용도와 시드 구성은 `results/per_image_scores/README.md` 참조.

---

## 4. test-free 프로토콜 — 무엇을 지켰나

**설정 선택에 test 를 한 번도 쓰지 않는다.** 판정은 두 기준으로만 했다.

| 기준 | 무엇을 재나 | 데이터 |
|---|---|---|
| train 변별력 | 분기가 논리 이상을 정상과 가르는 능력 | train 의사레이블 위 섭동 + LOO 1-NN AUROC |
| val 안정성 | z-정규화가 이상치에 견디는가 | val 정상 점수 꼬리의 \|z_robust\| 최댓값 |

**두 기준이 반대를 가리킬 수 있다.** val 정상 이미지 한 장이 표준편차를 부풀리면 그 분기의 test
z-점수가 통째로 줄어 사실상 꺼진다. 그래서 변별력만 봐서는 안 된다 — 실제로 반대를 가리킨
사례가 둘 있었고(k-NN k=5, 개수 거리), 둘 다 기각했다.

**기각한 것 중 수치가 가장 높았던 것도 기각했다**: h:c 가중(w_c=0.25)은 기준 평균 +0.0150 으로
최대 개선이었으나, 기준이 마스크를 조작하므로 면적항은 직접·외형항은 간접으로만 변해
**구조적으로 외형항을 과소평가한다**. 기준이 못 재는 것을 기준으로 채택할 수 없다.
전체 11건 중 채택 2 · 기각 9. 근거는 `results/diagnostics/PREREG_branch_improvement.md`.

> **읽을 때 오해하기 쉬운 지점**: `code/selection_criteria/soft_areas.py` 는 test 분할을
> 순회한다. 이는 **점수 계산**이지 판정이 아니다 — 그 실험(소프트 마스크)의 채택 여부는
> val 꼬리(pushpins 1.77 → 10.03)로 기각됐다. test AUROC 는 최종 1회 보고에만 쓴다.

---

## 5. 담기지 않은 것과 그 이유

| 항목 | 크기 | 재생성 |
|---|--:|---|
| DINOv3-L 특징 캐시 | ~18 GB | `code/branches/composition_psad/build_patchcore_pt.py` |
| UNet 분할 체크포인트 | ~9 GB | `code/branches/composition_psad/train_unet_seeds.sh` |
| EfficientAD 학습 가중치 | ~2 GB | `code/branches/reconstruction_ead/train_ead_seeds.sh` |
| MVTec LOCO AD 데이터셋 | ~6 GB | 원저작권자 배포 (MVTec) |
| PSAD 구성분기의 기각된 탐색 변형 점수 | — | `score_psad.py` 의 해당 플래그 |

**per-image 점수는 담았다** — 이것만으로 §1 의 검증이 되기 때문이다. 위 대용량은 그 점수를
처음부터 다시 만들려 할 때만 필요하다.

**ComAD 시드별 표준편차는 260831 에 해소했다** — 이전 판은 "per-image 점수가 이 머신에
없다" 고 적었으나 있었다(`external/ROMAD_baselines/ComAD/output_scores_seed{42,43,44}/`,
시드당 1,568개). `per_image_scores/comad/` 15개가 그것을 정본 순서로 묶은 것이고,
`results/tables/comad_axis_perseed.json` 이 축별 값이다. 산출 전에 재현부터 검증했다 —
기존 3시드 기록과 pooled 15조합, 정본과 축별 10칸 모두 **최대 절대차 0**.

산출 불가로 남은 것: human-annotation 변형(5범주×3시드가 없어 논문에서 해당 열을 제거했다).

**PUAD-S 시드 43 은 260821 에 해소했다** — 학습이 아니라 재채점으로 풀리는 문제였다.
`per_image_scores/puad_s/` 15개가 그 결과다.

---

## 6. 경로

스크립트는 **실제로 실행된 그대로**다 — 저장소 루트가 하드코딩돼 있다. 실행 기록을 사후
편집하지 않으려고 그대로 뒀다. 정확한 개수와 재지정 방법은 `code/RETARGET.md`.
`code/analysis/reproduce_from_package.py` 만은 절대경로가 없어 어디서든 돈다.

---

## 7. 무결성과 감사

`MANIFEST.md` 에 **파일마다 원본 저장소 경로와 SHA-256** 이 있다. 결측 파일이 있으면 조용히
넘기지 않고 MANIFEST 에 기재한다.

이 패키지는 조립 후 항목별 적대적 검수를 거쳤고, 그 과정에서 **결함 9건**이 나와 고쳤다.
검수는 `code/analysis/audit_submission_package.py` 로 재실행할 수 있다 — 8개 항목을 다시 돌린다.

| # | 결함 | 조치 |
|---|---|---|
| 1 | EAD 검증분할 점수 누락 — z-정규화 기준값이 없어 융합 재현 불가 | `per_image_scores/ead_s_val/` 추가 |
| 2 | PatchCore 분기 점수가 파일로 없었다 (18GB 캐시에서 즉석 계산) | 15개 내보내 `pc_branch/` 로 포함 |
| 3 | **`.gitignore` 의 `*.npz` 가 점수 파일 전체를 제외** — GitHub 에 올린 레포로는 아무것도 재계산되지 않는다 | `!submission/**/*.npz` 예외 추가 |
| 4 | `testfree_final_hcp_3seed.json` 이 stale — seed42 가 **원본 EAD 모델** 값이었다 (43·44 는 일치) | 재학습본으로 재생성 |
| 5 | 구성분기 npz 184개 중 인용되는 건 2계열뿐인데 기각 변형까지 전부 포함 | 인용 계열만 포함, 나머지는 재생성 절차 명시 |
| 6 | 표 JSON 에 **철회된 0.9807** 이 주 설정과 동등하게 나열 | `status: retracted` + 철회 사유 기재 |
| 7 | README 가 재빌드마다 소실 (`rmtree` 후 재생성) | 저장소 소스에서 복사하도록 변경 |
| 8 | 구성분기 종류(hc/hcp)가 산출물 config 에 없어 파일명으로만 구분됐다 | `psad_kind` 를 config 에 기록 |
| 9 | **PUAD-S 행이 3시드가 아니라 시드 44 단독**이었다 (집계 산출물이 나머지 시드 점수보다 먼저 굳었다). 게다가 구 시드 점수는 넷 다 체크포인트 출처를 신뢰할 수 없었다 — 셋은 다른 루트를 선언했고, 나머지 하나(44)는 선언한 루트를 실제로 보지 않았다 | {42,43,44} 를 현행 루트에서 재채점 → 0.9286 ± 0.0019. 검증은 EAD-S 교차대조로 (Spearman 1.000) |

검수 중 **자체 검증에 성공한 것**도 기록해 둔다: 내보낸 `pc_branch` 점수로 계산한 L+S 가
표 4 의 DINOv3-L 행과 **차이 0.000000**, SALAD·EAD-M 행도 0.000000, EAD-S·PUAD-M·PUAD-S 행은
소수점 6자리까지 일치했다. 즉 담긴 점수가 표를 만든 그 점수다.

**환경 재현성**: 이 패키지의 점수는 2026-05~06 에 산출됐고 PUAD-S 만 08 에 재산출됐다. 같은
체크포인트·같은 코드로 두 시점을 대조해 그 차이를 쟀다 — EAD-S 에서 **L+S −0.0006, Spearman
1.000, 시드 표준편차 불변**(`results/reproduction/puad_s_provenance_audit.json`). 부동소수
수준이며 표에 영향이 없다. §1 의 검증은 담긴 점수를 그대로 읽으므로 이 차이와 무관하다.

`docs/260716_v25_numbers_canonical.md` 는 **supersede 된 구 정본**이다(문서 첫머리에 대조표가
있다). 무엇이 왜 바뀌었는지가 기록이라 지우지 않고 남겼다. 현행 정본은
`docs/260817_testfree_hc_canonical.md`.

---

## 8. 데이터

MVTec LOCO AD (Bergmann et al., IJCV 2022). 학술 목적 무상 배포이며 재배포는 하지 않는다.
분할은 공식 train / validation / test 를 그대로 쓴다 — 재분할이나 test 유입이 없다.
