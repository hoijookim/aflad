# [논문 인계] 260822 — 조합 실험 히스토리 조사: 45건 있고, 거의 다 못 씁니다

**계기**: ④ 를 하다가 "AFLAD 조합에 이르기까지 여러 조합을 시험한 기록이 레포에 남아 있지
않나" 라는 물음이 나왔습니다. 전수 조사했습니다. **있습니다 — 아주 많이.** 다만 **현행
프로토콜로 쓸 수 있는 것은 사실상 없습니다.**

조사 방법: `reports/**/*.json` · `results/**/*.json` 에서 `method`/`name`/`note` 에 소스 조합을
가리키는 표현이 있는 것을 전수 추출 → **45건**.

---

## 1. 가장 직접적인 둘

### `reports/path_y/saturation/saturation_curve.json` — 31개 조합 전수

소스 5종 {v4, ead, pc, salad, puad} 의 모든 부분집합(2⁵−1 = 31)을 채점한 포화 곡선입니다.

| 크기 | 최고 조합 | AUROC | **SALAD 뺀 최고** |
|---|---|--:|--:|
| 1 | salad | 0.9383 | puad 0.9018 |
| 2 | salad+puad | 0.9378 | ead+puad 0.8950 |
| 3 | ead+salad+puad | 0.9386 | ead+pc+puad 0.9106 |
| 4 | ead+pc+salad+puad | 0.9441 | v4+ead+pc+puad 0.9175 |
| 5 | 전체 | 0.9471 | — |

**SALAD 단독(0.9383)이 SALAD 없는 모든 조합(최대 0.9175)을 이깁니다.**

### `reports/countgd/combo_sweep.json` — ④ 와 거의 같은 모양

V4+EAD+PC 를 기준으로 무엇을 더하면 얼마나 오르는지를 잰 sweep 입니다.

| 조합 | mean | Δ |
|---|--:|--:|
| baseline V4+EAD+PC | 0.9681 | — |
| +PUAD | 0.9697 | +0.16pp |
| +ComAD (SALAD-free comp) | 0.9734 | +0.53pp |
| +PUAD+ComAD (**SALAD-free best**) | 0.9741 | +0.60pp |
| **+SALAD_comp** | **0.9814** | **+1.33pp** |
| +PUAD+SALAD_comp | 0.9822 | +1.41pp |
| +PUAD+ComAD+SALAD_comp | 0.9829 | +1.48pp |

**구성 자리에 SALAD 를 넣는 것이 단일 추가 중 가장 컸습니다(+1.33pp).**

### 그 외

`direction_F` 5건(F_D5 4-src SALAD **+1.17pp**, F_D6 부트스트랩 CI, F_D7 5-src **−0.11pp**,
F_D8 per-cat dispatch, F_D10 subset dispatch) · `phase1b` 9건 · `phase2` 2건 · `phase3` 15건 ·
`direction_E/G/I` 5건 · `validation` 2건.

---

## 2. 왜 그대로 쓸 수 없나 — 세 가지가 겹칩니다

### (a) 정규화가 test 통계 기반입니다

`PY_saturation_curve.py`:

```python
def zscore(s, ref=None):
    if ref is None: ref = s          # ← test 점수 자신의 mean/std
    return (s - ref.mean()) / max(ref.std(), 1e-12)
```

호출부를 보면 **`ead` 만 `val_ref` 를 넘기고 `v4`·`pc`·`salad`·`puad` 넷은 `ref` 없이**
부릅니다. 즉 다섯 중 넷이 test 분포로 정규화됩니다. 단일 소스라면 AUROC 가 안 변하지만
**합산 융합에서는 소스 간 상대 가중이 바뀌므로 결과가 달라집니다.** `direction_F` 계열도
같은 패턴입니다(`zscore(v4_avg)`).

### (b) 중첩 교차검증 + 가중치 격자 — 260817 에 폐지한 프로토콜

`combo_sweep.py` 머리말: `inductive: fold별 train-good z-norm + nested CV`,
가중치 격자 `WG = [0.0, 0.2, 0.4, 0.6, 1.0]`. `direction_F`·`phase1b`·`phase3` 도 전부
`StratifiedKFold` 기반입니다. 주 설정은 **등가중 (1,1,1)** 이고 CV 를 쓰지 않습니다.

### (c) 지표가 다릅니다 — 값을 옮길 수 없습니다

포화 곡선은 5범주를 **한 덩어리로 묶어 pooled AUROC** 를 냅니다(`n_test: 1568`).
같은 소스를 정본과 대조하면 이렇습니다:

| 소스 | 포화 곡선 | 정본 |
|---|--:|--:|
| v4 | 0.7372 | pooled **0.9077** / L+S 0.9075 |
| ead | 0.8648 | L+S 0.8945 |
| pc | 0.7739 | L+S 0.7890 |
| salad | 0.9383 | L+S 0.9469 |

**v4 만 17pp 벌어집니다.** 범주별 AUROC 를 평균하는 것과 전 범주를 한 번에 채점하는 것은
다른 지표이고, 그 차이가 소스마다 다르게 나타납니다.

---

## 3. ④ 에 대해 이 히스토리가 말하는 것

### 방향 힌트로는 유효합니다

두 산출물이 독립적으로 같은 말을 합니다 — **구성 자리에 SALAD 를 넣으면 크게 오른다**
(combo_sweep +1.33pp, F_D5 +1.17pp). 포화 곡선에서도 SALAD 단독이 SALAD 없는 4-src 를
이깁니다. 요청서에 미리 적어 두신 갈래 중 **"A 가 비슷하거나 높으면"** 쪽이 나올 가능성이
있어 보입니다.

### 그러나 그 힌트는 생각보다 약합니다

**당시 구성 자리는 V4(DL-free)였고, 그 sweep 에서 가장 약한 소스였습니다**(포화 곡선 단독
0.7372 로 5종 중 최하위). 지금 우리 구성 분기는 PSAD-hc 로 **L+S 0.8948** 입니다.

즉 "+SALAD 가 +1.33pp" 는 **약한 구성 분기 위에서 잰 여유분**입니다. 지금은 그 자리가 이미
강하게 채워져 있어 같은 여유분이 남아 있지 않을 수 있습니다. ① 에서 나온 값이 이를 뒷받침합니다
— 우리 구성 분기의 **개수축 0.9926**, 논리 0.9505 로, 융합에서 그 자리가 맡는 몫이 큽니다.

**결론: 히스토리는 ④ 를 대신하지 못합니다.** SALAD 재학습(약 24시간)이 여전히 유일한 경로입니다.

---

## 4. 그래도 쓸 데가 있습니다

**"예전에 다 해보지 않았나" 라는 심사·지도 질문에 답이 됩니다.** 조합 탐색을 안 한 게 아니라
**했고, 프로토콜을 바꾸면서 그 결과를 버린 것**입니다. 그 사실 자체가 test-free 전환이
얼마나 광범위했는지를 보여줍니다 — 45건이 한 번에 무효가 됐습니다.

논문에 넣을 것을 권하지는 않습니다(폐기된 프로토콜의 수치를 본문에 들이면 설명 부담만 늘어납니다).
다만 **응답서에 쓸 재료로는 정확합니다.**

---

## 부록 — 현행 프로토콜로 된 조합 근거는 이것뿐입니다

| 산출물 | 무엇 |
|---|---|
| `testfree_supplements_3seed.json` | 우리 3분기의 단독 4 · 2소스 3조합 · 3소스 · 융합 방식 3종 |
| `homogeneous_ensemble_control.json` (②) | 동종 3× 대조군, 2-멤버 급, 부트스트랩 CI |
| `pc_backbone_swap.json` (④ 대체) | 패치 메모리 자리 5종 교체 |
| `branch_axis_decomposition.json` (①) | 분기 × 축 분해 |

**구성 자리 교체만 비어 있습니다.**
