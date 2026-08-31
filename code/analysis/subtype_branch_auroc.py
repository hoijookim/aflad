#!/usr/bin/env python3
"""요청 ⑫ — 하위유형 × **네 점수원** AUROC (융합·구성·재구성·패치 메모리), 5범주 3시드.

## 왜

`footprint_resolved_*.py` 는 **패치 메모리 하나만** 쟀다. 그래서 §5.2 가
"지금의 구성 분기로는 잡히지 않을 것으로 보인다" 라는 **예측**을 쓰고 있었다.
저자가 막았고, 요청 ⑫ 가 나머지 셋을 채워 달라고 한다.

논문이 주장해야 하는 것은 분기 하나가 아니라 **AFLAD 가 그 하위유형을 잡느냐**다.

## 정렬 — 이 작업의 유일한 위험

하위유형 라벨은 `test/logical_anomalies` 의 **파일명 순서**로 만들고, 점수는 npz 배열
순서로 온다. 둘이 어긋나면 AUROC 가 조용히 틀린다.

세 점수원의 정렬 근거가 서로 다르다:

| 점수원 | 순서 근거 |
|---|---|
| 구성(PSAD) | `paths` 배열이 있다 — **명시적** |
| 재구성(EAD-S) | `label_type` 배열이 있다 — 분할 경계만 확인 가능 |
| 패치 메모리(PC) | 배열뿐 — **근거 없음**, 정본 규약을 가정한다 |

그래서 **정본 순서를 직접 만들어**(good→logical→structural, 각 파일명 정렬) PSAD 의
`paths` 와 대조하고, EAD 의 `label_type` 과도 대조한다. 둘 다 맞으면 PC 도 같은 규약을
따른다고 볼 근거가 선다(길이·분할 경계가 일치하므로). 이 검증을 통과하지 못하면
값을 내지 않는다.

## 규약

- 하위유형: `defects_config.json` 의 pixel_value → defect_name. GT 마스크에 **단일**
  결함만 있는 이미지만 쓴다(MIXED 제외) — `footprint_resolved_allcat.py` 와 동일.
- 융합: 세 분기를 **검증 정상** 통계로 z-정규화한 뒤 등가중 합(§4.1 프로토콜).
- EAD-S seed42 는 `ead_s_seed42_repro`(패키지 README 규칙 — 정본 seed42 는 val 과
  출처가 섞여 융합이 이 파일을 쓴다). 43·44 는 `ead_s`.
- 구성은 `hc_tta_merge`(주 설정). `hcp_tta_merge_k5` 는 p 분기가 붙은 별도 설정이다.
- 3시드 {42,43,44} 평균±표준편차. n 을 함께 보고한다(요청 §2).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
L = R / "datasets/MVTecLOCO"
P = R / "submission/results/per_image_scores"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
SPLITS = [("good", "good"), ("logical_anomalies", "logical"),
          ("structural_anomalies", "structural")]
OUT = R / "reports/countgd/subtype_branch_auroc.json"
# 기존 patch-memory 값 — 교차 대조용(footprint_resolved_juice.json, 단일 실행)
PRIOR_JUICE_PC = {"wrong_juice_type": 0.5266, "misplaced_fruit_icon": 0.5947,
                  "missing_bottom_label": 0.6995, "swapped_labels": 0.7119,
                  "misplaced_label_top": 0.8383, "missing_fruit_icon": 0.8511}


def canonical(cat):
    """정본 순서 — good → logical → structural, 각 파일명 정렬."""
    order, types = [], []
    for d, t in SPLITS:
        for p in sorted((L / cat / "test" / d).glob("*.png")):
            order.append(f"test/{d}/{p.name}")
            types.append(t)
    return np.array(order), np.array(types)


def subtypes(cat):
    """logical 이미지별 하위유형. GT 마스크에 단일 결함만 있는 것만, 나머지는 MIXED."""
    cfg = {d["pixel_value"]: d["defect_name"]
           for d in json.load(open(L / cat / "defects_config.json"))}
    gt = L / cat / "ground_truth/logical_anomalies"
    out = []
    for p in sorted((L / cat / "test/logical_anomalies").glob("*.png")):
        g, pv = gt / p.stem, set()
        if g.is_dir():
            for m in g.glob("*.png"):
                pv |= set(np.unique(np.array(Image.open(m))).tolist())
        names = sorted(cfg[v] for v in pv if v in cfg)
        out.append(names[0] if len(names) == 1 else "MIXED")
    return np.array(out)


def z(t, v):
    return (t - v.mean()) / (v.std() + 1e-12)


def sources(cat, seed, paths, types):
    """네 점수원. 정렬을 검증한 뒤 반환한다 — 실패하면 예외로 세운다."""
    ps = np.load(P / f"psad_composition/psad_scores_{cat}_seed{seed}_hc_tta_merge_test.npz",
                 allow_pickle=True)
    pv = np.load(P / f"psad_composition/psad_scores_{cat}_seed{seed}_hc_tta_merge_val.npz",
                 allow_pickle=True)
    got = np.array([str(x).replace("\\", "/") for x in ps["paths"]])
    assert np.array_equal(got, paths), f"{cat}/s{seed}: PSAD paths 가 정본 순서와 다르다"

    ed = "ead_s_seed42_repro" if seed == 42 else "ead_s"
    e = np.load(P / f"{ed}/scores_{cat}_seed{seed}.npz", allow_pickle=True)
    lt = np.array([str(x) for x in e["label_type"]])
    assert np.array_equal(lt, types), f"{cat}/s{seed}: EAD label_type 이 정본과 다르다"
    ev = np.load(P / f"ead_s_val/val_good_{cat}_seed{seed}.npz", allow_pickle=True)

    pc = np.load(P / f"pc_branch/pc_L17_{cat}_seed{seed}.npz", allow_pickle=True)
    assert len(pc["test"]) == len(paths), f"{cat}/s{seed}: PC 길이 불일치"

    ze = z(np.asarray(e["score"], float), np.asarray(ev[ev.files[0]], float))
    zp = z(np.asarray(pc["test"], float), np.asarray(pc["val"], float))
    zc = z(np.asarray(ps["scores"], float), np.asarray(pv["scores"], float))
    return {"융합": ze + zp + zc, "구성": zc, "재구성": ze, "패치메모리": zp}


def main():
    rows, checked = {}, 0
    for cat in CATS:
        paths, types = canonical(cat)
        sub = subtypes(cat)
        gi = np.where(types == "good")[0]
        li = np.where(types == "logical")[0]
        si = np.where(types == "structural")[0]
        assert len(sub) == len(li), f"{cat}: 하위유형 {len(sub)} != logical {len(li)}"

        groups = {}
        for i, nm in enumerate(sub):
            if nm != "MIXED":
                groups.setdefault(nm, []).append(li[i])
        groups["__structural__"] = si.tolist()

        per = {nm: {k: [] for k in ("융합", "구성", "재구성", "패치메모리")} for nm in groups}
        for seed in SEEDS:
            sc = sources(cat, seed, paths, types)
            checked += 1
            for src, v in sc.items():
                for nm, pos in groups.items():
                    pos = np.array(pos)
                    per[nm][src].append(float(roc_auc_score(
                        np.r_[np.zeros(len(gi)), np.ones(len(pos))],
                        np.r_[v[gi], v[pos]])))
        rows[cat] = {nm: {"n": len(pos),
                          **{src: {"mean": float(np.mean(per[nm][src])),
                                   "std": float(np.std(per[nm][src], ddof=1)),
                                   "per_seed": per[nm][src]}
                             for src in per[nm]}}
                     for nm, pos in groups.items()}

    print(f"\n정렬 검증 {checked}/{len(CATS)*len(SEEDS)} 조합 통과 "
          f"(PSAD paths·EAD label_type 을 정본 순서와 대조)\n")

    for cat in CATS:
        print(f"=== {cat} ===")
        print(f"{'하위유형':<32}{'n':>4}{'융합':>16}{'구성':>16}{'재구성':>16}{'패치메모리':>16}")
        for nm, v in sorted(rows[cat].items(), key=lambda kv: kv[1]["융합"]["mean"]):
            lab = "구조이상(전체)" if nm == "__structural__" else nm
            cells = "".join(f"{v[s]['mean']:>10.4f}±{v[s]['std']:.3f}"
                            for s in ("융합", "구성", "재구성", "패치메모리"))
            print(f"{lab:<32}{v['n']:>4}{cells}")
        print()

    # 기존 patch-memory 단일 실행값과 대조 — 같은 대상을 재는지 확인
    print("=== 교차 대조: juice_bottle 패치메모리 vs footprint_resolved_juice.json ===")
    print(f"{'하위유형':<28}{'기존(1실행)':>12}{'이번 3시드':>14}{'차':>10}")
    for nm, old in PRIOR_JUICE_PC.items():
        if nm in rows["juice_bottle"]:
            new = rows["juice_bottle"][nm]["패치메모리"]["mean"]
            print(f"{nm:<28}{old:>12.4f}{new:>14.4f}{new-old:>+10.4f}")

    OUT.write_text(json.dumps({
        "question": "하위유형별로 네 점수원이 어떻게 갈리는가 (요청 ⑫)",
        "protocol": "good vs 각 순수 하위유형 AUROC · 검증정상 z-정규화 등가중 합 · 3시드",
        "seeds": SEEDS, "alignment_verified": checked, "per_cat": rows},
        ensure_ascii=False, indent=2))
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
