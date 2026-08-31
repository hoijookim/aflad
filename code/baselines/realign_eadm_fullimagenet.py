#!/usr/bin/env python3
"""표 5 의 EAD-M 을 full ImageNet 학습분으로 재산출한다 (시드 42/43/44).

## 결함
표 5 의 EAD-M(0.7846)은 `reports/phase0/efficient_ad_medium`(anomalib, **imagenette 폴백**)
에서 왔다. EfficientAD 의 penalty 항은 ImageNet 이미지로 student 의 과잉 일반화를 막는데,
imagenette(10클래스 소규모)로 대체하면 그 효과가 크게 약해진다. 같은 표의 EAD-S 는
`train_ead_seeds.sh` 에서 full ImageNet 을 썼으므로 조건이 다르고, 그 결과 출판 순서
(EAD-M 90.7 > EAD-S 90.0)가 뒤집혀 있다.

## 출처
full ImageNet 학습분은 `efficient_ad_official_medium*` 이다. 체크포인트는 원본 머신에서
소실됐으나 **시드별 이미지 점수가 살아 있다**:
  seed42 : reports/phase0/efficient_ad_official_medium/npz/scores_{cat}_seed42.npz
  42/43/44: reports/path_y/puad_m_multiseed_scores/{cat}_seed{S}.npz 의 `ead` 열
            (PY_real_puad_M_MULTISEED.py 가 official_medium 체크포인트로 산출)

교차검증: seed42 에서 두 산출물의 상관 0.9999~1.0000, L+S 소수 3자리 일치 — 같은 모델이다.
그래서 3시드 모두 `ead` 열로 일관되게 산출한다(정본 npz 와 섞지 않는다).

test 라벨은 이미 발표된 베이스라인을 학습 설정만 바로잡아 재집계하는 데만 쓴다.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
SC = R / "reports/path_y/puad_m_multiseed_scores"
NPZ = R / "reports/phase0/efficient_ad_official_medium/npz"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
TARGET = (42, 43, 44)


def main():
    # 1) 출처 교차검증 (seed42)
    print("출처 교차검증 (seed42): official_medium 정본 npz vs PUAD ead 열")
    ok = True
    for c in CATS:
        a = np.load(NPZ / f"scores_{c}_seed42.npz", allow_pickle=True)
        b = np.load(SC / f"{c}_seed42.npz", allow_pickle=True)
        n = min(len(a["score"]), len(b["ead"]))
        r = float(np.corrcoef(a["score"].astype(float)[:n], b["ead"].astype(float)[:n])[0, 1])
        print(f"  {c:22s} 상관 {r:.4f}")
        ok &= r > 0.999
    print(f"  => {'PASS 같은 모델' if ok else 'FAIL 다른 모델 — 중단'}\n")
    if not ok:
        return

    # 2) 3시드 재집계
    per = {}
    for sd in TARGET:
        lg, st = [], []
        for c in CATS:
            z = np.load(SC / f"{c}_seed{sd}.npz", allow_pickle=True)
            k = np.load(NPZ / f"scores_{c}_seed42.npz", allow_pickle=True)
            lab, lt = k["label"].astype(int), k["label_type"].astype(str)
            assert (z["labels"].astype(int) == lab).all(), f"{c}/seed{sd} 라벨 불일치"
            s = z["ead"].astype(float)
            for acc, kind in ((lg, "logical"), (st, "structural")):
                m = (lt == "good") | (lt == kind)
                acc.append(roc_auc_score(lab[m], s[m]))
        per[sd] = {"LS": float(0.5 * (np.mean(lg) + np.mean(st))),
                   "logical": float(np.mean(lg)), "structural": float(np.mean(st))}
        print(f"  EAD-M(full ImageNet) seed{sd}: L+S {per[sd]['LS']:.4f}")
    a = np.array([per[s]["LS"] for s in TARGET])
    out = {"EAD-M (full ImageNet)": {
        "per_seed": {str(k): v for k, v in per.items()},
        "LS_mean": float(a.mean()), "LS_std": float(a.std(ddof=1)),
        "logical": float(np.mean([per[s]["logical"] for s in TARGET])),
        "structural": float(np.mean([per[s]["structural"] for s in TARGET])),
        "supersedes": {"path": "reports/phase0/efficient_ad_medium", "LS": 0.7846,
                       "reason": "imagenette 폴백으로 penalty 정칙화가 약해짐"}}}
    print(f"\n  => EAD-M (full ImageNet) {a.mean():.4f} ± {a.std(ddof=1):.4f}  "
          f"(log {out['EAD-M (full ImageNet)']['logical']:.4f} / "
          f"str {out['EAD-M (full ImageNet)']['structural']:.4f})")
    print(f"     구 표기(imagenette) 0.7846 ± 0.0106 대비 {a.mean()-0.7846:+.4f}")
    p = R / "reports/countgd/realign_eadm_fullimagenet_3seed.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
