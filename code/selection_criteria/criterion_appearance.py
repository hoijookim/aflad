#!/usr/bin/env python3
"""외형 섭동 기준 — 지금까지 측정 불가였던 영역을 연다. train 전용, test 미조회.

## 왜 필요한가
기존 기준(criterion_v2 / criterion_exact)은 **마스크만** 조작한다. 그래서:
  - juice_bottle 의 실제 논리 이상(주스 종류 바뀜, 라벨 오류)은 **마스크가 전혀 변하지
    않는** 외형 이상이라 모사 자체가 불가능했다. 이 범주의 -1.2pp 는 측정 불가 영역이었다.
  - E8b(h:c 가중)를 기각한 이유도 이것이다 — 마스크 기준은 c 항을 구조적으로 과소평가한다.

여기서는 **픽셀을 바꾸고 마스크는 그대로 둔다**. 면적 히스토그램(h)은 불변이므로
이 섭동은 c 항만 시험한다 — 정확히 지금까지 못 재던 것이다.

## 섭동 유형 (전부 train 이미지 안에서 닫힌다)
  recolor   클래스 k 픽셀을 **같은 이미지의 다른 클래스** 평균색으로 치환
            (juice 종류 바뀜 / 부품 오종에 대응)
  desat     클래스 k 픽셀을 회색조로 (색 정보 소실)
  texswap   클래스 k 픽셀을 **다른 학습 이미지의 같은 클래스** 픽셀로 치환
            (같은 부품의 다른 개체 — 가장 현실적이나 정렬이 필요해 근사)

외부 데이터를 끌어오지 않고 test 도 보지 않는다.

## 판정
정상 train 의 LOO 1-NN 거리 vs 섭동 이미지의 1-NN 거리 AUROC, probe 시드 5개.
h 는 불변이므로 c 항의 변별력이 그대로 드러난다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
PSAD_DIR = R / "external/PSAD_official"
sys.path.insert(0, str(PSAD_DIR))
import psad as P  # noqa: E402

DATA = PSAD_DIR / "LOCO_MVTec_AD"
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}
MERGE = {"screw_bag": [[4, 5]]}          # E5 채택안과 일치
PROBES = [0, 1, 2, 3, 4]
KINDS = ("recolor", "desat", "texswap")


class Args:
    input_size = 512
    avgpool_size = 5


def load_lab(cat, seed, name):
    lab = np.array(Image.open(DATA / f"unet_seed{seed}" / cat / "train/good" / name))
    for g in MERGE.get(cat, []):
        g = sorted(g)
        for j in g[1:]:
            lab[lab == j] = g[0]
    return lab


@torch.no_grad()
def row_from(enc, img_rgb, lab, K):
    """공식 규약대로 [면적(K) | 클래스특징((K-1)*1792)] 한 행."""
    im = (img_rgb / 255. - P.imagenet_mean) / P.imagenet_std
    x = torch.tensor(np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)).float().unsqueeze(0).cuda()
    fts = enc(x)[0]                                    # [1792,H,W]
    C = fts.shape[0]
    F2 = fts.reshape(C, -1).t()                        # [HW,1792]
    area = np.zeros(K); out = np.zeros((K, C))
    flat = lab.ravel()
    for k in range(1, K):
        idx = np.nonzero(flat == k)[0]
        area[k] = len(idx)
        if len(idx):
            sel = torch.from_numpy(idx).cuda()
            out[k] = F2.index_select(0, sel).sum(0).cpu().numpy()
    del fts, F2
    ft = out[1:] / (area[1:, None] + 1)
    return np.concatenate([area, ft.ravel()])


def perturb(img, lab, k, kind, rng, donor=None):
    """마스크는 그대로 두고 클래스 k 의 픽셀만 바꾼다."""
    m = lab == k
    if not m.any():
        return None
    out = img.copy()
    if kind == "recolor":
        others = [v for v in np.unique(lab) if v not in (0, k)]
        if not others:
            return None
        o = others[rng.integers(len(others))]
        out[m] = img[lab == o].mean(0).astype(img.dtype)
    elif kind == "desat":
        g = img[m].mean(1, keepdims=True)
        out[m] = np.repeat(g, 3, axis=1).astype(img.dtype)
    else:                                              # texswap
        if donor is None:
            return None
        dimg, dlab = donor
        dm = dlab == k
        if not dm.any():
            return None
        src = dimg[dm]
        idx = rng.integers(0, len(src), size=int(m.sum()))
        out[m] = src[idx]
    return out


@torch.no_grad()
def evaluate(cat, seed=42, n_probe=60):
    K = NUM_CLS[cat]
    enc = P.Encoder(avgpool_size=Args.avgpool_size).cuda().eval()
    img_dir = DATA / "orig_512" / cat / "train/good"
    names = sorted(p.name for p in img_dir.glob("*.png"))
    imgs, labs, rows = [], [], []
    for i, n in enumerate(names):
        img = np.array(Image.open(img_dir / n).convert("RGB"))
        lab = load_lab(cat, seed, n)
        imgs.append(img); labs.append(lab)
        rows.append(row_from(enc, img, lab, K))
        if i % 60 == 0:
            torch.cuda.empty_cache()
            print(f"    {cat} 정상 {i}/{len(names)}", flush=True)
    M = np.stack(rows)
    mu, sd = M.mean(0), M.std(0) + 1e-10
    Z = (M - mu) / sd

    def nn(q, ex):
        d = np.sqrt(((Z - q) ** 2).sum(1)); d[ex] = np.inf
        return d.min()

    normal = np.array([nn(Z[i], i) for i in range(len(Z))])
    out = {}
    for kind in KINDS:
        vals = []
        for ps in PROBES:
            rng = np.random.default_rng(ps); pert = []
            for i in rng.choice(len(Z), min(n_probe, len(Z)), replace=False):
                cand = [k for k in range(1, K) if (labs[i] == k).sum() > 200]
                if not cand:
                    continue
                k = cand[rng.integers(len(cand))]
                donor = None
                if kind == "texswap":
                    j = int(rng.integers(len(Z)))
                    if j == i:
                        continue
                    donor = (imgs[j], labs[j])
                pi = perturb(imgs[i], labs[i], k, kind, rng, donor)
                if pi is None:
                    continue
                q = (row_from(enc, pi, labs[i], K) - mu) / sd
                pert.append(nn(q, i))
            if len(pert) < 10:
                break
            vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                      np.r_[normal, pert]))
        out[kind] = (float(np.mean(vals)), float(np.std(vals, ddof=1))) if len(vals) > 1 else None
        if out[kind]:
            print(f"    {cat} {kind:9s} {out[kind][0]:.4f} ± {out[kind][1]:.4f}", flush=True)
    del enc
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(NUM_CLS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-probe", type=int, default=60)
    a = ap.parse_args()
    res = {}
    for cat in a.cats.split(","):
        print(f"\n=== {cat} (외형 섭동, 마스크 불변)", flush=True)
        res[cat] = {k: v for k, v in evaluate(cat, a.seed, a.n_probe).items()}
    print("\n" + "=" * 70)
    print(f"{'범주':22s} " + " ".join(f"{k:>16s}" for k in KINDS))
    for cat, r in res.items():
        print(f"{cat:22s} " + " ".join(
            (f"{r[k][0]:9.4f}±{r[k][1]:.3f}" if r.get(k) else f"{'--':>16s}") for k in KINDS))
    p = R / "reports/countgd/criterion_appearance.json"
    json.dump(res, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
