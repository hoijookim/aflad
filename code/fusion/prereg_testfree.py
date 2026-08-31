#!/usr/bin/env python3
"""test-free 사전 등록(pre-registration) 융합 — 설계 선택을 val/train 통계로만 확정.

배경: 앞선 시도(0.9807)는 정규화·클리핑·집계를 **test AUROC 로 비교해 골랐다**.
점수 계산은 test-free 였지만 구성 선택이 test 를 봤으므로 그 수치는 test 로 튜닝된
값이고 test-free 라 부를 수 없다. 여기서는 그 오염을 제거한다.

## 결정 규칙 (test 를 보기 전에 확정, 이 파일에 고정)

D1 분기 구성 — 아키텍처 근거로 고정, 성능 비교 없음.
   ead + pcL17 + psad_hc. hcp 는 구성분기 내부 PatchCore 가 독립 PC 분기와 중복되어
   "세 분기 직교" 전제를 깨므로 제외(260630 설계 결정).

D2 스케일 — 세 분기 점수는 모두 양수·비율척도·상한없음(EAD val 최소 0.023 / PC 95.3 /
   PSAD 0.393)이다. 이런 양에 선형 표준화를 적용하면 비대칭이 남는다. val 정상 분포의
   평균 |왜도| 로 스케일을 고른다: 원점수 1.542 / log1p 1.329 / **log 0.879** -> log 채택.

D3 표준화 — log 스케일에서 robust(중앙값/IQR). val 최대 |z_robust| 가 22.21 로
   정규분포 기대치(n≈60 에서 ≈2.0)를 크게 벗어나 평균/표준편차는 부적합하다.
   클리핑은 두지 않는다 — 임계값을 정할 val 근거가 없고(최대값은 단일 이상치가
   지배하는 불안정한 추정량), log 가 이미 꼬리를 압축한다.

D4 집계 — 논문 기본값 유지: 단순 합.
   val 에는 이상 표본이 없어 집계 방식을 고를 근거가 없다. 근거 없이 바꾸지 않는다.

D5 시드 — 보유 PSAD 시드 전부. EAD·PC 는 같은 시드로 짝지어 함께 변동시킨다.

## 실행
  1) val/train 진단 -> D2·D3 값 확정 및 출력
  2) 확정된 구성으로 test 1회 평가
  중간에 test 를 보고 되돌아가지 않는다.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import fuse_testfree_v2 as F  # noqa: E402

REB = R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
BRANCHES = ["ead", "pcL17", "psad_hc"]          # D1 고정


# 재학습 EAD 의 seed42 test 점수. 정본 npz(scores_*_seed42.npz)는 **원본 모델**의 것이라
# 재학습 val 과 짝지으면 z-정규화 기준이 다른 모델에서 온다(두 모델 상관 0.89~0.997).
# seed43~46 은 정본 경로가 이미 재학습본이므로 seed42 만 이 경로로 대체해 출처를 통일한다.
EAD_REPRO = R / "reports/phase0/ead_repro_npz"


@lru_cache(maxsize=None)
def ead_one(cat, seed):
    tp = EAD_REPRO / f"scores_{cat}_seed{seed}.npz"
    if not tp.exists():
        tp = F.EAD_T / f"scores_{cat}_seed{seed}.npz"
    t = np.load(tp, allow_pickle=True)
    v = np.load(F.EAD_V / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
    return (t["score"].astype(float), v["scores"].astype(float),
            t["label"].astype(int), t["label_type"].astype(str))


@lru_cache(maxsize=None)
def pc_one(cat, seed):
    """DINOv3 L17 PatchCore. 5만 패치로 kNN 을 새로 세우므로 범주당 수십 초가 든다 —
    진단 루프에서 같은 (범주, 시드)를 여러 번 부르므로 캐시가 필수다."""
    c = np.load(R / "cache/dinov3_multilayer_vitl16" / f"{cat}.npz")
    Xtr = c["train_L17"].astype("f4"); Xte = c["test_L17"].astype("f4")
    nv = len(list((R / "datasets/MVTecLOCO" / cat / "validation" / "good").glob("*.png")))
    ntr = Xtr.shape[0] - nv
    npt, d = Xtr.shape[1], Xtr.shape[2]
    fl = Xtr[:ntr].reshape(-1, d)
    idx = np.random.default_rng(seed).choice(fl.shape[0], min(50000, fl.shape[0]), replace=False)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(fl[idx])

    def sc(X):
        di, _ = nn.kneighbors(X.reshape(-1, d))
        return di.mean(1).reshape(X.shape[0], npt).max(1)

    return sc(Xte), sc(Xtr[ntr:])


PSAD_TAG = ""          # "" = 현행(standardize=1), "_std0" = E1, "_merge" = E5
PSAD_KIND = "hc"       # hc(헤드라인 분기 구성) / hcp(절제 비교용)


@lru_cache(maxsize=None)
def psad_one(cat, seed):
    t = np.load(REB / f"psad_scores_{cat}_seed{seed}_{PSAD_KIND}{PSAD_TAG}_test.npz", allow_pickle=True)
    v = np.load(REB / f"psad_scores_{cat}_seed{seed}_{PSAD_KIND}{PSAD_TAG}_val.npz", allow_pickle=True)
    return t["scores"].astype(float), v["scores"].astype(float)


BRANCH_AVG_SEEDS = None      # None 이면 시드별, 리스트면 그 시드들의 평균으로 고정


def branch(b, cat, seed):
    if BRANCH_AVG_SEEDS is not None and b in ("ead", "pcL17"):
        # 원 스크립트가 EAD·PC 를 시드평균으로 고정한 것을 그대로 재현한다.
        ts, vs = [], []
        for sd in BRANCH_AVG_SEEDS:
            t, v = (ead_one(cat, sd)[:2] if b == "ead" else pc_one(cat, sd))
            ts.append(t); vs.append(v)
        return np.mean(ts, 0), np.mean(vs, 0)
    if b == "ead":
        t, v, _, _ = ead_one(cat, seed)
        return t, v
    if b == "pcL17":
        return pc_one(cat, seed)
    return psad_one(cat, seed)


def robust_z(x, v):
    med = np.median(v)
    iqr = max(float(np.subtract(*np.percentile(v, [75, 25]))), 1e-9)
    return (x - med) / iqr


def main():
    global PSAD_TAG, PSAD_KIND
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--psad-tag", default="", choices=["", "_std0", "_soft", "_merge", "_tta_merge", "_tta_merge_k5", "_merge_k5", "_std0_soft"],
                    help="PSAD 채점 변형 태그 (E1: _std0, E4: _soft)")
    ap.add_argument("--branch-avg", action="store_true",
                    help="원본 프로토콜 정합: EAD·PC 를 시드평균으로 고정하고 PSAD 만 변동")
    ap.add_argument("--kind", default="hc", choices=["hc", "hcp"],
                    help="hc=헤드라인 분기 구성, hcp=절제 비교용")
    ap.add_argument("--out", default=None)
    ap.add_argument("--protocol", default="paper", choices=["paper", "diagnostic"],
                    help="paper: 논문이 선언한 test-free 정의 그대로 "
                         "(검증 정상 z-정규화 + 등가중 1,1,1 + 합). "
                         "diagnostic: val 통계로 스케일·표준화를 고르는 변형(이탈)")
    a = ap.parse_args()
    PSAD_TAG = a.psad_tag
    PSAD_KIND = a.kind
    global BRANCH_AVG_SEEDS

    pat = f"psad_scores_*_{PSAD_KIND}{PSAD_TAG}_test.npz"
    seeds = sorted({int(p.name.split("_seed")[1].split("_")[0])
                    for p in REB.glob(pat)})
    seeds = [s for s in seeds
             if all((REB / f"psad_scores_{c}_seed{s}_{PSAD_KIND}{PSAD_TAG}_test.npz").exists()
                    for c in F.CATS)]
    print(f"PSAD 구성: {PSAD_KIND} / 변형: {PSAD_TAG or '현행'}")
    if a.branch_avg:
        BRANCH_AVG_SEEDS = tuple(seeds)
        print(f"분기 평균 모드: EAD·PC 를 시드 {seeds} 평균으로 고정, PSAD 만 변동")
    print(f"D5 시드: {seeds}\n")
    if not seeds:
        print("완비된 시드 없음 — 종료")
        return

    # ---------- 단계 1: 정규화 확정 ----------
    print("=" * 74)
    if a.protocol == "paper":
        print("단계 1 — 논문 선언 프로토콜 그대로 (고를 것이 없다)")
        print("=" * 74)
        print("논문 §3.2/§5.6 정의: 「검증(정상) 표본만으로 z-정규화하고 가중을 고정한 구성」,")
        print("고정 가중 = 등가중 (1,1,1). 스케일 변환·robust 통계·클리핑은 정의에 없다.")
        print("  D2 스케일   = 원점수")
        print("  D3 표준화   = z (검증 정상 평균/표준편차), 클리핑 없음")
        print("  D4 집계     = 등가중 합")
        SCALE, NORM, CLIP = "raw", "z", None
        g = lambda x: x  # noqa: E731
        tails = {}
        for cat in F.CATS:
            for b in BRANCHES:
                v = branch(b, cat, seeds[0])[1]
                tails[(cat, b)] = float(np.max(np.abs(robust_z(v, v))))
    else:
        print("단계 1 — val-good 진단으로 정규화·클리핑 확정 (test 미조회)")
        print("=" * 74)
        SCALE = NORM = CLIP = g = tails = None
    # D2 — val 정상 분포의 왜도로 스케일을 고른다 (원점수 / log1p / log)
    def skew(x):
        x = np.asarray(x, float); s = x.std()
        return 0.0 if s < 1e-12 else float((((x - x.mean()) / s) ** 3).mean())

    if a.protocol == "diagnostic":
        SCALES = {"raw": lambda x: x,
                  "log1p": lambda x: np.log1p(np.maximum(x, 0)),
                  "log": lambda x: np.log(np.maximum(x, 1e-12))}
        sk = {}
        for nm, f_ in SCALES.items():
            sk[nm] = float(np.mean([abs(skew(f_(branch(b, cat, seeds[0])[1])))
                                    for cat in F.CATS for b in BRANCHES]))
        SCALE = min(sk, key=sk.get)
        print("D2 스케일 — val 정상 분포 평균 |왜도|: " +
              " / ".join(f"{k} {v:.3f}" for k, v in sk.items()) + f"  -> {SCALE}")

        # D3 — 선택된 스케일에서 꼬리 통계로 표준화 방식을 고른다. 클리핑은 두지 않는다
        #      (임계값을 정할 val 근거가 없고, 최대값은 단일 이상치가 지배하는 불안정한 추정량).
        g = SCALES[SCALE]
        tails = {}
        for cat in F.CATS:
            for b in BRANCHES:
                v = g(branch(b, cat, seeds[0])[1])
                tails[(cat, b)] = float(np.max(np.abs(robust_z(v, v))))
        print(f"\n{'범주':22s} " + " ".join(f"{b:>9s}" for b in BRANCHES) +
              f"   ← {SCALE} 스케일 val 최대 |z_robust|")
        for cat in F.CATS:
            print(f"{cat:22s} " + " ".join(f"{tails[(cat,b)]:9.2f}" for b in BRANCHES))
        T = max(tails.values())
        NORM = "robust" if T > 4 else "z"
        CLIP = None
        print(f"\n판정통계 T = 최대 {T:.2f}  (정규분포 n≈60 기대치 ≈ 2.0)")
        print(f"  D3 표준화 = {NORM}   (규칙: T > 4 이면 중앙값/IQR)")
        print(f"  D3 클리핑 = 없음 (임계값을 정할 val 근거가 없다)")
        print(f"  D4 집계 = 단순 합 (논문 기본값, 변경 근거 없음)")

    # ---------- 단계 2: 확정 구성으로 test 1회 평가 ----------
    print("\n" + "=" * 74)
    print("단계 2 — 확정 구성으로 test 1회 평가")
    print("=" * 74)

    def norm(t, v):
        t, v = g(t), g(v)
        if NORM == "z":
            return (t - v.mean()) / max(v.std(), 1e-9)
        return robust_z(t, v)

    per_seed = {}
    for sd in seeds:
        per_cat = {}
        for cat in F.CATS:
            _, _, lab, lt = ead_one(cat, sd)
            Z = np.stack([norm(*branch(b, cat, sd)) for b in BRANCHES])
            f = Z.sum(0)                                   # D4
            lg = roc_auc_score(lab[(lt == "good") | (lt == "logical")],
                               f[(lt == "good") | (lt == "logical")])
            st = roc_auc_score(lab[(lt == "good") | (lt == "structural")],
                               f[(lt == "good") | (lt == "structural")])
            per_cat[cat] = {"logical": lg, "structural": st, "LS": 0.5 * (lg + st)}
        agg = {k: float(np.mean([per_cat[c][k] for c in F.CATS]))
               for k in ("logical", "structural", "LS")}
        per_seed[sd] = {"per_cat": per_cat, "aggregate": agg}
        print(f"  seed {sd}: L+S {agg['LS']:.4f} "
              f"(log {agg['logical']:.4f} / str {agg['structural']:.4f})")

    out = {"config": {"branches": BRANCHES, "scale": SCALE, "norm": NORM, "clip": CLIP,
                      "agg": "sum", "seeds": seeds,
                      "decision_rule": "D2: min mean abs-skew of val ; D3: T>4 -> robust, no clip ; D4: sum"},
           "val_tail_stat": {f"{c}|{b}": tails[(c, b)] for c in F.CATS for b in BRANCHES},
           "per_seed": {str(k): v["aggregate"] for k, v in per_seed.items()},
           "per_cat": {c: {"mean": float(np.mean([per_seed[s]["per_cat"][c]["LS"] for s in seeds])),
                           "std": (float(np.std([per_seed[s]["per_cat"][c]["LS"] for s in seeds], ddof=1))
                                   if len(seeds) > 1 else None)} for c in F.CATS}}
    for k in ("logical", "structural", "LS"):
        v = np.array([per_seed[s]["aggregate"][k] for s in seeds])
        out[k] = {"mean": float(v.mean()),
                  "std": (float(v.std(ddof=1)) if len(v) > 1 else None)}
    out["config"]["psad_variant"] = PSAD_TAG or "standardize=1 (현행)"
    # 구성분기 종류를 산출물에 남긴다 — 파일명만으로는 hc/hcp 를 구분할 수 없었다.
    out["config"]["psad_kind"] = PSAD_KIND
    out["config"]["branch_avg"] = bool(a.branch_avg)  # 시드평균 여부도 남긴다
    out["config"]["branches"] = ["ead", "pcL17", f"psad_{PSAD_KIND}"]
    op = Path(a.out) if a.out else (
        R / f"reports/countgd/testfree_prereg{PSAD_TAG}.json")
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)

    s_ = out["LS"]["std"]
    print(f"\n사전 등록 구성 test-free L+S = {out['LS']['mean']:.4f}"
          + (f" ± {s_:.4f}" if s_ is not None else " (단일 시드)"))
    print(f"  logical {out['logical']['mean']:.4f} / structural {out['structural']['mean']:.4f}")
    print(f"  범주별: " + " ".join(f"{c[:5]} {out['per_cat'][c]['mean']:.3f}" for c in F.CATS))
    print(f"[saved] {op}")


if __name__ == "__main__":
    main()
