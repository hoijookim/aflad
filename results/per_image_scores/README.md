# per-image 점수 — 무엇을 뒷받침하나

각 npz 는 이미지 단위 이상 점수다. 이 파일들만으로 헤드라인이 재계산된다 (`code/analysis/reproduce_from_package.py`). 캐시·체크포인트는 필요 없다.

| 디렉터리 | 파일 | 시드 | 용도 |
|---|--:|---|---|
| `comad` | 15 | 42, 43, 44 |  |
| `ead_m_imagenette` | 15 | 42, 43, 44 | EAD-M 의 imagenette 폴백 학습본. 표 4 에서 교체된 수치의 원본 증거. |
| `ead_s` | 25 | 0, 42, 43, 44, 1234 | EAD-S 재구성 분기 test 점수. 표 4 의 EAD-S 행과 융합의 첫째 항. |
| `ead_s_seed42_repro` | 5 | 42 | seed42 재학습본 test 점수. 정본 seed42 는 원본 모델 산출물이라 val(재학습본)과 출처가 섞였다. 융합은 이 파일을 쓴다. |
| `ead_s_val` | 15 | 42, 43, 44 | EAD-S 검증분할 정상 표본 점수. z-정규화의 기준값 — 없으면 융합이 재현되지 않는다. |
| `original_pipeline_reference` | 50 | 0, 42, 1234 | 원본 PSAD/PatchCore 파이프라인 참조 점수 (재구축 대조군). |
| `pc_branch` | 15 | 42, 43, 44 | DINOv3-L L17 PatchCore 분기 (coreset 50000 / k=1 / max / train-only 뱅크). 표 4 의 DINOv3-L 행과 융합의 둘째 항이 같은 값이다. |
| `psad_composition` | 60 | 42, 43, 44 | PSAD 구성분기. hc_tta_merge = 주 설정, hcp_tta_merge_k5 = 부록 비교. |
| `puad_m` | 15 | 42, 43, 44 | PUAD-M 재현 (시드별 EAD-M + 공식 Mahalanobis). npz 에 유형 배열이 없다 — test 이미지 순서가 `ead_s` 와 동일함을 이진 라벨 배열 일치로 확인했으므로 `label_type` 을 거기서 빌려 쓴다(재현 스크립트가 그렇게 한다). |
| `puad_s` | 15 | 42, 43, 44 | PUAD-S 재현 (EAD-S 체크포인트 + 공식 Mahalanobis, 학습 없이 재채점). 구 산출물은 시드마다 체크포인트 루트가 달라 셋 다 `_seeds_std` 에서 다시 매겼다. |
| `salad` | 15 | 42, 43, 44 | SALAD 재현 (자체 학습 3시드). |

## 시드에 관해

주 표는 **{42, 43, 44}** 다. `ead_s` 의 45·46 은 부록의 EAD-S 5시드 재현(`results/reproduction/ead_5seed_verdict.json`), 0·1234 는 원본 PSAD 프로토콜(0/42/1234)과 맞춘 대조 실행에 쓰인다.

## 담지 않은 것

PSAD 구성분기의 기각된 탐색 변형(k=5·개수거리·소프트마스크 등 14계열)은 제외했다. 기각 근거는 `results/diagnostics/PREREG_branch_improvement.md` 와 `sweep_*.json` 에 있고, 점수 자체는 `code/branches/composition_psad/score_psad.py` 의 해당 플래그로 재생성된다.
