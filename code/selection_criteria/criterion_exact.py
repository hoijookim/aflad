#!/usr/bin/env python3
"""채택 후보를 **공식 특징 경로 위에서 정확히** 재검증한다.

왜 필요한가 — 자체 검수에서 발견한 것:
  psad_scoring_search.py 의 캐시는 공식 경로와 두 곳이 다르다.
    (1) 공식 Encoder.forward 는 layer1~3 에 avg_pool2d(k=5,pad=2,stride=1) 평활화를
        건 뒤 512 로 bilinear 업샘플해 concat(1792d) 한다. 프록시는 평활화가 없다.
    (2) 프록시는 반대로 마스크를 특징 해상도로 내려 곱한다.
  두 변형(std=1/0)이 같은 캐시를 쓰므로 A/B 비교 자체는 공정하지만, 결론이 실제
  파이프라인으로 옮겨가는지는 별개 문제다. 여기서 그것을 확인한다.

정확성 확보 방법 — **연결성분 단위 캐시**:
  공식 클래스 특징은 ft_cls = (fts * mask_cls).sum() / (mask_cls.sum() + 1) 이다.
  이는 클래스에 속한 픽셀들의 특징 **합**을 면적+1 로 나눈 것이므로, 성분별로
  (픽셀수, 특징합) 만 저장해두면 성분을 빼거나 다른 클래스로 옮긴 뒤의 클래스 특징을
  **재추론 없이 정확히** 재구성할 수 있다.
    area[k]  = sum(cnt[c] for c in k에 속한 성분)
    feat[k]  = sum(fsum[c] for c in k에 속한 성분) / (area[k] + 1)
  remove(성분 제거)와 swap(성분 클래스 변경)은 이 재구성으로 오차 없이 모사된다.
  (duplicate/resize 는 새 픽셀의 특징이 필요해 재추론 없이는 정확하지 않으므로 뺀다.)

입력은 train/good 뿐이다. test 는 어느 단계에서도 쓰지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
# score_psad.confused_pairs 가 train 혼동행렬로 자동 검출한 결과 (seed42, thr=0.10).
# 여기 하드코딩하는 이유는 기준 스크립트가 채점기를 import 하면 순환이 되기 때문이고,
# 값 자체는 train 만 보고 나온 것이다.
MERGE_GROUPS = {"screw_bag": [[4, 5]]}
PSAD_DIR = R / "external/PSAD_official"
sys.path.insert(0, str(PSAD_DIR))
import psad as P  # noqa: E402

DATA = PSAD_DIR / "LOCO_MVTec_AD"
CACHE = R / "reports/countgd/psad_exact_cache"
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}
PROBES = [0, 1, 2, 3, 4]
MIN_COMP = 200


class Args:
    input_size = 512
    avgpool_size = 5


@torch.no_grad()
def build_cache(cat, seed=42, merge=False):
    """이미지별 연결성분 (클래스, 픽셀수, 특징합[1792])."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{cat}_seed{seed}{'_merge' if merge else ''}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return list(z["cls"]), list(z["cnt"]), list(z["fsum"])
    num_cls = NUM_CLS[cat]
    enc = P.Encoder(avgpool_size=Args.avgpool_size).cuda().eval()
    seg_root = DATA / f"unet_seed{seed}" / cat / "train/good"
    img_root = DATA / "orig_512" / cat / "train/good"
    paths = sorted(img_root.glob("*.png"))
    CLS, CNT, FSUM = [], [], []
    for i, p in enumerate(paths):
        lab = np.array(Image.open(seg_root / p.name))
        comp, n = ndimage.label(lab > 0)
        sizes = ndimage.sum(lab > 0, comp, range(1, n + 1)) if n else []
        keep = [j + 1 for j, s in enumerate(sizes) if s >= MIN_COMP]
        if not keep:
            CLS.append(np.zeros(0, np.int64)); CNT.append(np.zeros(0, np.float32))
            FSUM.append(np.zeros((0, 1792), np.float32))
            continue
        img = P.read_img(Args(), str(p))
        x = torch.tensor(img).float().cuda().unsqueeze(0).permute(0, 3, 1, 2)
        fts = enc(x)[0]                                  # [1792,512,512] (공식 경로)
        C = fts.shape[0]
        F2 = fts.reshape(C, -1)                          # [1792,HW]
        M = torch.from_numpy(
            np.stack([(comp == c).ravel() for c in keep])).float().cuda()   # [nc,HW]
        fsum = (M @ F2.t()).cpu().numpy()                # [nc,1792] 정확한 픽셀 합
        cnt = M.sum(1).cpu().numpy()
        cls = np.array([int(np.bincount(lab[comp == c]).argmax()) for c in keep])
        if merge:
            for g in MERGE_GROUPS.get(cat, []):
                g = sorted(g)
                for j in g[1:]:
                    cls[cls == j] = g[0]
        CLS.append(cls); CNT.append(cnt.astype(np.float32)); FSUM.append(fsum.astype(np.float32))
        del fts, F2, M
        if i % 40 == 0:
            torch.cuda.empty_cache()
            print(f"    {cat} {i}/{len(paths)}", flush=True)
    np.savez(f, cls=np.array(CLS, dtype=object), cnt=np.array(CNT, dtype=object),
             fsum=np.array(FSUM, dtype=object))
    del enc
    torch.cuda.empty_cache()
    return CLS, CNT, FSUM


def memory_row(cls, cnt, fsum, num_cls, drop=None, swap=None):
    """공식 규약 그대로 [면적(num_cls) | 클래스특징(1792*(num_cls-1))] 한 행을 만든다."""
    # 성분 개수가 모든 이미지에서 같으면 np.array(..., dtype=object) 가 3차원 object
    # 배열이 되어 float 누산이 실패한다. 여기서 dtype 을 확정해 그 경우를 흡수한다.
    cls = np.asarray(cls, dtype=np.int64)
    cnt = np.asarray(cnt, dtype=np.float64)
    fsum = np.asarray(fsum, dtype=np.float64)
    if fsum.ndim == 1:
        fsum = fsum.reshape(len(cls), -1)
    c = cls.copy()
    keep = np.ones(len(c), bool)
    if drop is not None:
        keep[drop] = False
    if swap is not None:
        c[swap[0]] = swap[1]
    area = np.zeros(num_cls, np.float64)
    fts = np.zeros((num_cls, fsum.shape[1] if fsum.size else 1792), np.float64)
    for j in range(len(c)):
        if not keep[j]:
            continue
        area[c[j]] += cnt[j]
        fts[c[j]] += fsum[j]
    ft = fts[1:] / (area[1:, None] + 1)                  # 공식: /(mask.sum()+1)
    return np.concatenate([area, ft.ravel()])


def evaluate(cat, standardize, kind, seed=42, n_probe=120, w_c=1.0, k=1, merge=False):
    CLS, CNT, FSUM = build_cache(cat, seed, merge)
    K = NUM_CLS[cat]
    M = np.stack([memory_row(CLS[i], CNT[i], FSUM[i], K) for i in range(len(CLS))])
    if standardize:
        mu, sd = M.mean(0), M.std(0) + 1e-10
        Z = (M - mu) / sd
    else:
        mu, sd = np.zeros(M.shape[1]), np.ones(M.shape[1]); Z = M

    def dist(q, ex):
        d = (Z - q) ** 2
        h = np.sqrt(d[:, :K].sum(1)); c = np.sqrt(d[:, K:].sum(1))
        h[ex] = np.inf; c[ex] = np.inf
        if k == 1:
            return h.min(), c.min()
        # k-NN 평균 — 메모리 항목 하나가 점수를 결정하는 취약성을 줄인다
        return np.sort(h)[:k].mean(), np.sort(c)[:k].mean()

    hs, cs = map(np.array, zip(*[dist(Z[i], i) for i in range(len(Z))]))
    hn, cn = max(hs.max(), 1e-10), max(cs.max(), 1e-10)   # 공식 scale_type=max
    normal = hs / hn + w_c * cs / cn

    vals = []
    for ps in PROBES:
        rng = np.random.default_rng(ps)
        pert = []
        for i in rng.choice(len(Z), min(n_probe, len(Z)), replace=False):
            nc = len(CLS[i])
            if nc == 0:
                continue
            j = int(rng.integers(nc))
            if kind == "remove":
                q = memory_row(CLS[i], CNT[i], FSUM[i], K, drop=j)
            else:
                labs = [v for v in np.unique(CLS[i]) if v != CLS[i][j] and v != 0]
                if not labs:
                    continue
                q = memory_row(CLS[i], CNT[i], FSUM[i], K,
                               swap=(j, int(labs[rng.integers(len(labs))])))
            q = (q - mu) / sd
            h, c = dist(q, i)
            pert.append(h / hn + w_c * c / cn)
        if len(pert) < 10:
            return None, None
        vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                  np.r_[normal, pert]))
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(NUM_CLS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = {}
    print(f"{'범주':22s} {'유형':8s} {'std=1(현행)':>16s} {'std=0':>16s} {'차이':>9s}")
    for cat in a.cats.split(","):
        out[cat] = {}
        for kind in ("remove", "swap"):
            r1 = evaluate(cat, 1, kind, a.seed)
            r0 = evaluate(cat, 0, kind, a.seed)
            if r1[0] is None or r0[0] is None:
                print(f"{cat:22s} {kind:8s} {'--':>16s}"); continue
            out[cat][kind] = {"std1": r1, "std0": r0, "diff": r0[0] - r1[0]}
            print(f"{cat:22s} {kind:8s} {r1[0]:9.4f}±{r1[1]:.4f} {r0[0]:9.4f}±{r0[1]:.4f} "
                  f"{r0[0]-r1[0]:+9.4f}", flush=True)
    for kind in ("remove", "swap"):
        d = [out[c][kind]["diff"] for c in out if kind in out[c]]
        if d:
            print(f"\n{kind}: 평균 {np.mean(d):+.4f}, 개선 {sum(x>0 for x in d)}/{len(d)}")
    p = R / "reports/countgd/criterion_exact.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
