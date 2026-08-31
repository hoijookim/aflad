#!/usr/bin/env python3
"""PatchCore 계열 3종을 **train-only 메모리 뱅크**로 통일 재산출 (시드 42/43/44).

## 왜 바꾸는가
기존 `PY_ls_patchcore_family*.py` 는 뱅크를 `train_L{layer}` **전체**로 만든다. 그런데 이
캐시는 train + validation 을 함께 담고 있다(예: screw_bag 420 = train 360 + val 60).
즉 **validation 이미지가 모델 데이터로 뱅크에 들어간다**.

세 가지가 걸린다:
1. **§4.1 과 충돌** — "모든 기준선을 동일한 데이터 분할과 평가 코드로 재실행"이라 선언해 두고
   PatchCore 만 val 을 모델 데이터로 쓰면 그 선언이 깨진다.
2. **다른 기준선과 원칙이 어긋난다** — EfficientAD·SALAD 는 val 을 정규화 통계 산출에만 쓰고
   모델 데이터로 넣지 않는다. PatchCore 뱅크만 예외가 된다.
3. **우리 융합의 PC 분기는 val 을 반드시 빼야 한다** — val 이 뱅크에 있으면 각 val 이미지가
   자기 자신을 최근접으로 찾아 거리 ≈ 0 이 되고, 그 val 통계로 test 를 z-정규화하는 test-free
   프로토콜이 망가진다. 그래서 뱅크 정의가 갈리면 같은 방법이 표 안에서 두 값을 갖는다.

**"val 을 넣는 쪽이 기준선에 유리하다"는 것도 사실이 아니다** — 실측에서 오히려 낮다
(DINOv3-L 0.7879 < train-only 0.7890). `cs=50000` 캡 때문에 뱅크 용량은 늘지 않고 샘플링만
달라지며, 표준편차가 0.0064 대 0.0018 로 3.5배인 것이 성능 차가 아니라 **코어셋 추출 노이즈**
임을 가리킨다.

=> 세 변형 모두 train-only 로 통일한다. 그러면 표 4 의 DINOv3-L PatchCore 행과 표 5 의
   patch-memory 분기 열이 **하나의 값**이 되고 각주가 필요 없다.

## 계산
하이퍼파라미터는 기존과 동일: coreset 50000, k=1, agg=max, 레이어 DINOv2-B/DINOv3-B=11,
DINOv3-L=17. 바뀌는 것은 뱅크에서 validation 을 빼는 것 하나뿐이다.
라벨·결함유형은 V4 정본(`algml_v6_5/aupr_bootstrap/scores`)에서 가져온다.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

R = Path("/workspace/ai-vision-research")
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = (42, 43, 44)
CS, K, AGG = 50000, 1, "max"
MODELS = {
    "DINOv2-B": ("cache/dinov2_multilayer_vitb14", 11),
    "DINOv3-B": ("cache/dinov3_multilayer_vitb16", 11),
    "DINOv3-L": ("cache/dinov3_multilayer_vitl16", 17),
}


def labels(cat):
    z = np.load(R / "algml_v6_5/aupr_bootstrap/scores" / f"scores_{cat}_seed42.npz",
                allow_pickle=True)
    return z["label"].astype(int), z["label_type"].astype(str)


def n_val(cat):
    return len(list((R / "datasets/MVTecLOCO" / cat / "validation" / "good").glob("*.png")))


def score(cache_dir, layer, cat, pc_seed, train_only=True):
    c = np.load(R / cache_dir / f"{cat}.npz")
    Xtr = c[f"train_L{layer}"].astype("f4")
    Xte = c[f"test_L{layer}"].astype("f4")
    npt, d = Xtr.shape[1], Xtr.shape[2]
    if train_only:
        ntr = Xtr.shape[0] - n_val(cat)          # 캐시는 train + val 을 함께 담는다
        Xtr = Xtr[:ntr]
    flat = Xtr.reshape(-1, d)
    if CS < flat.shape[0]:
        idx = np.random.default_rng(pc_seed).choice(flat.shape[0], CS, replace=False)
        bank = flat[idx]
    else:
        bank = flat
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1).fit(bank)
    di, _ = nn.kneighbors(Xte.reshape(-1, d))
    ps = di.mean(1).reshape(Xte.shape[0], npt)
    return ps.max(1) if AGG == "max" else ps.mean(1)


def trio(s, lab, lt):
    lg = roc_auc_score(lab[(lt == "good") | (lt == "logical")],
                       s[(lt == "good") | (lt == "logical")])
    st = roc_auc_score(lab[(lt == "good") | (lt == "structural")],
                       s[(lt == "good") | (lt == "structural")])
    return {"pooled": float(roc_auc_score(lab, s)), "logical": float(lg),
            "structural": float(st), "LS": float(0.5 * (lg + st))}


def main():
    LAB = {c: labels(c) for c in CATS}
    out = {"config": {"bank": "train-only (validation 제외)", "coreset": CS, "k": K,
                      "agg": AGG, "seeds": list(SEEDS)}, "models": {}}
    print(f"{'모델':10s} {'시드':>5s} {'L+S':>8s}   (train-only 뱅크)")
    for m, (cdir, layer) in MODELS.items():
        per = {}
        for sd in SEEDS:
            t = []
            for c in CATS:
                s = score(cdir, layer, c, sd)
                t.append(trio(s, *LAB[c]))
            per[sd] = {k: float(np.mean([x[k] for x in t]))
                       for k in ("pooled", "logical", "structural", "LS")}
            print(f"{m:10s} {sd:5d} {per[sd]['LS']:8.4f}", flush=True)
        a = np.array([per[s]["LS"] for s in SEEDS])
        out["models"][m] = {
            "per_seed": {str(k): v for k, v in per.items()},
            "LS_mean": float(a.mean()), "LS_std": float(a.std(ddof=1)),
            "logical": float(np.mean([per[s]["logical"] for s in SEEDS])),
            "structural": float(np.mean([per[s]["structural"] for s in SEEDS])),
            "pooled": float(np.mean([per[s]["pooled"] for s in SEEDS]))}
        print(f"  => {m}: L+S {a.mean():.4f} ± {a.std(ddof=1):.4f}  "
              f"(log {out['models'][m]['logical']:.4f} / "
              f"str {out['models'][m]['structural']:.4f})\n", flush=True)

    # 교차검증: DINOv3-L 이 융합의 PC 분기와 일치해야 한다 (같은 계산이므로)
    import sys
    sys.path.insert(0, str(R / "scripts/psad_rebuild"))
    import prereg_testfree as PT
    vals = []
    for sd in SEEDS:
        t = [trio(PT.pc_one(c, sd)[0], *LAB[c]) for c in CATS]
        vals.append(float(np.mean([x["LS"] for x in t])))
    v = np.array(vals)
    d3l = out["models"]["DINOv3-L"]["LS_mean"]
    print(f"교차검증 — 융합의 PC 분기: {v.mean():.4f} ± {v.std(ddof=1):.4f}")
    print(f"           재산출 DINOv3-L: {d3l:.4f}   차이 {abs(v.mean()-d3l):.6f}  "
          f"{'PASS 동일 계산' if abs(v.mean()-d3l) < 1e-4 else 'FAIL 계산 불일치'}")
    out["crosscheck_fusion_pc_branch"] = {"LS_mean": float(v.mean()),
                                          "LS_std": float(v.std(ddof=1)),
                                          "diff_vs_DINOv3-L": float(abs(v.mean() - d3l))}
    p = R / "reports/countgd/patchcore_family_trainonly_3seed.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
