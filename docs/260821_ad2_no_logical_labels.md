# [논문 인계] 260821 — MVTec AD 2 에는 논리 하위유형이 없다 (§5.2 향후 과제 · [11])

**결론: A 로 진행하십시오. 되살릴 근거가 없습니다.**

부록 C 의 분해(논리 하위유형 45개를 국소 흔적 유무로 가름)를 MVTec AD 2 에서 반복하는 것은
**원칙적으로 불가능**합니다. AD 2 에는 논리/구조 라벨이 없는 정도가 아니라 **결함 유형 라벨
자체가 없습니다.** 추정이 아니라 우리가 그 데이터셋을 실제로 다룰 때 쓴 코드가 증거입니다.

---

## 1. AD 2 의 이상은 폴더 하나에 뭉쳐 있다

`env_robust_ad/common/dataset.py` (v5 AD 2 실험의 공식 로더)가 순회하는 전부:

```
{category}/train/good
{category}/validation/good
{category}/test_public/good
{category}/test_public/bad                  ← 이상은 전부 여기 한 폴더
{category}/test_public/ground_truth/bad
```

`env_robust_ad/phase0/baseline_dinomaly.py:106` 은 유형을 이렇게 기록합니다:

```python
types.append('bad')      # 단일 리터럴. 분기가 없다.
```

**LOCO 와 대조하면 차이가 분명합니다.** 이 머신의 `datasets/MVTecLOCO/breakfast_box/test/` 는
`good` / `logical_anomalies` / `structural_anomalies` 세 폴더입니다 — 부록 C 의 분해는 이
층 위에 서 있습니다. AD 2 에는 그 층이 존재하지 않습니다.

## 2. AD 2 가 이미지 단위로 주는 유일한 축은 결함이 아니라 **촬영 조건**이다

같은 로더의 파일명 파서:

```python
# 조명 조건 파싱: {id}_{condition}.png
_FILENAME_RE = re.compile(r"^(\d+)_(.+)\.png$")

condition: str   # regular, overexposed, underexposed, shift_1, ...
```

GT 마스크도 `{id}_{condition}_mask.png` 이고, 데이터 클래스에 있는 편의 메서드는
`is_regular` / `is_shifted` / `test_by_condition(condition)` 입니다. **결함 의미를 가르는
메서드는 없습니다.** 초록이 말하는 난점(조명 변화·분포 이동)과 정확히 일치하는 설계입니다.

## 3. 우리 AD 2 산출물에도 유형 분해가 없다

v5 결과 JSON 전수를 확인했습니다. `logical` / `structural` / `defect` / `type` 키가
**한 건도 없습니다.** 지표 축은 `mean_*_AP` / `mean_*_SF`(SegF1) / 범주별 값입니다
(예: `env_robust_ad/v5/results_b3_4encoder_fusion.json`). 유형별로 나눌 수 있었다면
어딘가에 남았을 텐데 없습니다.

CLAUDE.md 에 `fruit_jelly … logical anomaly`, `vial … logical scoring` 같은 표현이 있는 것은
**우리 v5 실험의 작업 용어**이지 데이터셋 분류가 아니라는 논문 세션의 판단이 맞습니다.

## 4. 원본 확인도 물리적으로 불가능하고, 해도 결과가 같다

이 머신에 AD 2 원본과 특징 캐시가 **둘 다 없습니다** — `datasets/` 에는 `MVTecAD` 와
`MVTecLOCO` 만 있고 `features/` 디렉터리는 존재하지 않습니다. tar.gz 삭제(2026-05-24)와
별개로 8/15 백업 복원 때 대용량이 빠진 것으로 보입니다.

**다시 내려받아도 결과는 같습니다.** 위 §1~2 는 그 데이터셋을 실제로 읽던 코드이고, 폴더
구조는 데이터셋이 정한 것이지 우리 선택이 아니기 때문입니다.

## 5. 그래서 두 주장이 바꿔치기된 것이 맞다

| 주장 | 근거 상태 |
|---|---|
| "AD 2 는 어렵다" (구조 탐지 AU-PRO < 60%) | 초록에서 확인됨. **단, 원인은 조명·분포 이동** |
| "AD 2 는 흔적 축의 시험대다" | **뒷받침 불가** — 흔적 축을 재려면 논리 하위유형 라벨이 필요한데 없다 |

전자가 후자를 지지하지 않습니다. 데이터셋이 어려운 이유와 우리가 재려는 축이 다릅니다.

## 6. 권고

**A — 지금 빼고 [11] 재번호까지 진행.** 보류할 이유가 없습니다. "확인이 오면 되살린다"의
그 확인이 방금 끝났고, 되살릴 수 없다는 답입니다.

제안하신 대체 문장은 그대로 성립하며, 오히려 이 조사가 그 문장을 정확하게 만듭니다:

> 둘째, 국소 흔적을 남기는지가 탐지 난이도를 가른다는 관찰(부록 C)은 MVTec LOCO 한
> 데이터셋에서 얻은 것이다. **논리 이상 하위유형이 라벨된** 다른 벤치마크에서 같은 분해를
> 반복하면 이 축이 데이터셋에 특유한 것인지 가려낼 수 있다.

굵게 표시한 조건절이 핵심입니다 — AD 2 는 정확히 그 조건을 만족하지 않는 데이터셋이고,
따라서 문장이 AD 2 를 지목하지 않는 것이 **누락이 아니라 정확한 것**입니다.

## 7. 참고 — 조건이 맞는 후보

지금 지목할 필요는 없지만, 나중에 이 문장을 구체화할 때 필요한 것은 논리 하위유형이 라벨된
벤치마크입니다. 우리가 다룬 것 중에는 **MVTec LOCO 뿐**입니다.

- **MVTec AD** — 이 머신에서 확인했습니다. `bottle/test/` 가 `broken_large` / `broken_small` /
  `contamination` / `good` 으로, 유형 폴더는 있으나 전부 **외형 수준 결함**이고 논리 축이 없습니다.
- **VisA** — 이 머신에 원본이 없어 확인하지 못했습니다(코드가 `datasets/VisA/split_csv/1cls.csv`
  를 참조하지만 디렉터리가 없습니다). 다만 §3.3 교차검증 때 우리가 쓴 경로가 1-class split
  이므로 유형 분해를 쓴 적은 없습니다.

새 데이터셋 확보가 필요한 작업이므로 향후 과제로 두는 것이 맞습니다.
