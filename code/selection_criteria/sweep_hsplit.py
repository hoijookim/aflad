#!/usr/bin/env python3
"""h 항의 개수 신호를 **거리 수준에서 분리**해 훑는다 — train 전용, test 미조회.

## 동기
breakfast_box 의 duplicate 탐지가 0.514(우연)다 — 클래스별 면적만으로는 부품이 하나
늘어난 것과 기존 부품이 조금 커진 것을 구분할 수 없다. E2 에서 성분 개수를 면적 벡터에
**이어붙여** 봤지만 기각됐다: 차원이 늘면서 L2 거리에서 면적 신호가 희석된다.

여기서는 붙이지 않고 **거리를 따로 만든다**:
    s = h_area/scale_a + w_n * h_cnt/scale_n + c/scale_c
각 항을 자기 스케일(train LOO 최댓값)로 나눈 뒤 더하므로 차원 희석이 없다.
같은 이유로 c 항도 건드리지 않는다 — E8 에서 확인했듯 이 기준은 마스크 조작 기반이라
c 를 과소평가하므로, c 의 가중은 1.0 에 고정한다.

## 기준
criterion_exact 의 성분단위 캐시로 remove/swap 을 오차 없이 모사한 LOO 거리 AUROC.
다만 **duplicate 는 이 캐시로 정확히 모사할 수 없다**(새 픽셀의 특징이 필요). 대신
성분 하나를 복제해 같은 클래스의 면적·개수·특징합을 더하는 근사로 둔다 — 면적과 개수가
정확히 바뀌므로 h 항 평가에는 충분하고, c 는 가중평균이 되어 실제와 가깝다.

## 채택 규칙(선언, test 를 보기 전에 고정)
5범주 x 3유형(remove/swap/duplicate) 평균이 현행(w_n=0) 대비 +0.005 이상이고,
악화 범주의 낙폭이 그 범주 현행값의 1% 이내일 것.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import criterion_exact as CE  # noqa: E402

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PROBES = [0, 1, 2, 3, 4]
KNN = 5                      # E8 채택안과 동일


def rows(cat, seed, merge):
    CLS, CNT, FSUM = CE.build_cache(cat, seed, merge)
    return [np.asarray(c, np.int64) for c in CLS], \
           [np.asarray(c, np.float64) for c in CNT], \
           [np.asarray(f, np.float64).reshape(len(CLS[i]), -1) if len(CLS[i]) else
            np.zeros((0, 1792)) for i, f in enumerate(FSUM)]


def descriptors(cls, cnt, fsum, K, drop=None, swap=None, dup=None):
    """(면적[K], 개수[K], 클래스특징[(K-1)*1792]) 를 공식 규약대로 만든다."""
    c = cls.copy()
    keep = np.ones(len(c), bool)
    if drop is not None:
        keep[drop] = False
    if swap is not None:
        c[swap[0]] = swap[1]
    area = np.zeros(K); num = np.zeros(K)
    fts = np.zeros((K, fsum.shape[1] if fsum.size else 1792))
    for j in range(len(c)):
        if not keep[j]:
            continue
        area[c[j]] += cnt[j]; num[c[j]] += 1; fts[c[j]] += fsum[j]
    if dup is not None:                      # 성분 하나 복제 (개수 +1, 면적 +cnt)
        k = c[dup]
        area[k] += cnt[dup]; num[k] += 1; fts[k] += fsum[dup]
    return area, num, (fts[1:] / (area[1:, None] + 1)).ravel()


def evaluate(cat, w_n, seed=42, merge=True, n_probe=120):
    CLS, CNT, FSUM = rows(cat, seed, merge)
    K = CE.NUM_CLS[cat]
    D = [descriptors(CLS[i], CNT[i], FSUM[i], K) for i in range(len(CLS))]
    A = np.stack([d[0] for d in D]); N = np.stack([d[1] for d in D])
    C = np.stack([d[2] for d in D])

    def z(X):
        mu, sd = X.mean(0), X.std(0) + 1e-10
        return (X - mu) / sd, mu, sd

    Za, mua, sda = z(A); Zn, mun, sdn = z(N); Zc, muc, sdc = z(C)
    ka = Za.std(0) > 1e-8; kn = Zn.std(0) > 1e-8
    Za, Zn = Za[:, ka], Zn[:, kn]

    def d1(Z, q, ex):
        v = np.sqrt(((Z - q) ** 2).sum(1)); v[ex] = np.inf
        return v.min() if KNN == 1 else np.sort(v)[:KNN].mean()

    def trip(i):
        return (d1(Za, Za[i], i), d1(Zn, Zn[i], i) if Zn.shape[1] else 0.0, d1(Zc, Zc[i], i))

    T = np.array([trip(i) for i in range(len(Za))])
    sc = np.maximum(T.max(0), 1e-10)
    normal = T[:, 0] / sc[0] + w_n * (T[:, 1] / sc[1] if Zn.shape[1] else 0) + T[:, 2] / sc[2]

    out = {}
    for kind in ("remove", "swap", "duplicate"):
        vals = []
        for ps in PROBES:
            rng = np.random.default_rng(ps); pert = []
            for i in rng.choice(len(Za), min(n_probe, len(Za)), replace=False):
                nc = len(CLS[i])
                if nc == 0:
                    continue
                j = int(rng.integers(nc))
                if kind == "remove":
                    a, n_, f = descriptors(CLS[i], CNT[i], FSUM[i], K, drop=j)
                elif kind == "duplicate":
                    a, n_, f = descriptors(CLS[i], CNT[i], FSUM[i], K, dup=j)
                else:
                    labs = [v for v in np.unique(CLS[i]) if v != CLS[i][j] and v != 0]
                    if not labs:
                        continue
                    a, n_, f = descriptors(CLS[i], CNT[i], FSUM[i], K,
                                           swap=(j, int(labs[rng.integers(len(labs))])))
                qa = ((a - mua) / sda)[ka]; qn = ((n_ - mun) / sdn)[kn]
                qc = (f - muc) / sdc
                s = (d1(Za, qa, i) / sc[0] + w_n * (d1(Zn, qn, i) / sc[1] if Zn.shape[1] else 0)
                     + d1(Zc, qc, i) / sc[2])
                pert.append(s)
            if len(pert) < 10:
                break
            vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                      np.r_[normal, pert]))
        out[kind] = float(np.mean(vals)) if vals else None
    got = [v for v in out.values() if v is not None]
    return (float(np.mean(got)) if got else None), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cats = a.cats.split(",")
    grid = [0.0, 0.25, 0.5, 1.0, 2.0]
    res = {}
    for w in grid:
        row = {}
        for cat in cats:
            m, det = evaluate(cat, w, a.seed)
            row[cat] = m
        res[str(w)] = row
        got = [v for v in row.values() if v is not None]
        mark = "  <= 현행(개수 미사용)" if w == 0.0 else ""
        print(f"w_n={w:<5} 평균 {np.mean(got):.4f}  " +
              " ".join(f"{c[:5]} {row[c]:.4f}" for c in cats) + mark, flush=True)

    base = res["0.0"]
    print("\n채택 규칙: 평균 +0.0050 이상 & 악화 낙폭이 현행값의 1% 이내")
    ok = []
    for w in grid[1:]:
        r = res[str(w)]
        d = {c: r[c] - base[c] for c in cats}
        g = float(np.mean(list(d.values())))
        wc = min(d, key=d.get)
        st = "통과" if (g >= 0.005 and d[wc] >= -0.01 * base[wc]) else "미달"
        print(f"  {st} w_n={w:<5} 평균 {g:+.4f}  최악 {wc[:5]} {d[wc]:+.4f}")
        if st == "통과":
            ok.append((g, w))
    if not ok:
        print("  -> 현행(개수 미사용) 유지")
    p = R / "reports/countgd/sweep_hsplit.json"
    json.dump({"results": res, "accepted": [w for _, w in sorted(ok, reverse=True)]},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
