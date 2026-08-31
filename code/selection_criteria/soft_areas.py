#!/usr/bin/env python3
"""UNet 분할의 하드 argmax 를 소프트 확률로 완화하고, 그 효과를 val 로만 검증한다.

## 문제 (diagnose_rebuild_gap.py C 단계에서 특정)
PSAD 의 h 항은 클래스별 면적 히스토그램인데, 그 면적을 argmax 분할에서 센다.
argmax 는 모델 확신에 대해 불연속이다 — 두 부품 클래스 사이가 51:49 인 화소와
49:51 인 화소가 완전히 다른 클래스로 떨어진다. screw_bag val 029 가 정확히 그
사고다: cls5 가 0 픽셀이 되고 cls4 가 2.6배(train 1926+-85 -> 4940, 35 sigma)로
부풀어 PSAD 점수가 |z_robust| 22.21 까지 튀었다. 정상 이미지인데도 그렇다.

## 완화
히스토그램이 추정하려는 양은 "각 부품이 차지하는 면적"이다. 예측 분포 아래의
**기대 면적** sum_pixel softmax_k 가 그 양의 자연스러운 연속 추정량이고, 모델이
두 유사 부품 사이에서 흔들릴 때 질량을 나눠 가지므로 완만하게 열화한다.
c 항의 클래스별 특징 평균도 같은 소프트 가중으로 일관되게 확장한다.

## 검증 (train + val 만, test 미조회)
지표: val 정상 이미지의 클래스 면적 벡터가 train 분포에서 몇 sigma 벗어나는가.
정상 이미지는 train 분포 안에 있어야 하므로, 이상치 수와 최대 편차가 줄면 개선이다.
채택 규칙(선언): 5범주 합계 이상치(>5 sigma) 수가 줄고, 어느 범주에서도 최대 편차가
늘지 않을 것.

산출: LOCO_MVTec_AD/unet_seed{seed}_soft/{cat}/{split}/{type}/NNN.npz  (area[num_cls])
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

REPO = Path("/workspace/ai-vision-research/external/PSAD_official")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

DATA = REPO / "LOCO_MVTec_AD"
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}
USE_COORD = {"breakfast_box": True, "juice_bottle": True, "pushpins": True,
             "screw_bag": True, "splicing_connectors": True}
SPLITS = [("train", "good"), ("validation", "good"), ("test", "good"),
          ("test", "logical_anomalies"), ("test", "structural_anomalies")]
IMNET_MEAN = np.array([0.485, 0.456, 0.406])
IMNET_STD = np.array([0.229, 0.224, 0.225])


def load_model(cat, seed, level=3):
    # CNNSegmenter 는 train_normal_unet 안에 정의돼 있다. import 하면 argparse 가 도는
    # main() 은 실행되지 않으므로(모듈 최상단이 아니다) 안전하다.
    import train_normal_unet as T
    m = T.CNNSegmenter(num_cls=NUM_CLS[cat], use_coord=USE_COORD[cat],
                       level=level, pretrained=False).cuda().eval()
    sd = torch.load(DATA / f"output/unet_seed{seed}/{cat}/{cat}_300.pth", map_location="cpu")
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def run_cat(cat, seed):
    """공식 ValDataSet2D 를 그대로 써서 전처리·coord 규약을 추측하지 않는다."""
    from dataset_2d_sup import ValDataSet2D
    from torch.utils.data import DataLoader

    m = load_model(cat, seed)
    ds = ValDataSet2D(root=str(DATA), obj_name=cat, size=(512, 512),
                      save_dir=f"unet_seed{seed}_soft")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    out_root = DATA / f"unet_seed{seed}_soft" / cat
    out_root.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for b in dl:
        x = b["image"].cuda(); coord = b["coord_orig"].cuda()
        opath = Path(b["opath"][0])
        logit = m(x, coord)
        prob = Fn.softmax(logit, dim=1)[0]                     # [K,H,W]
        hard = torch.zeros_like(prob)
        hard.scatter_(0, logit[0].argmax(0, keepdim=True), 1.0)
        key = (opath.parent.parent.name, opath.parent.name)    # (split, type)
        buckets.setdefault(key, ([], [], []))
        buckets[key][0].append(hard.sum((1, 2)).cpu().numpy())
        buckets[key][1].append(prob.sum((1, 2)).cpu().numpy())
        buckets[key][2].append(opath.name)
    stats = {}
    for (split, typ), (h, s, n) in buckets.items():
        H, S = np.stack(h), np.stack(s)
        np.savez(out_root / f"{split}_{typ}.npz", hard=H, soft=S, names=np.array(n))
        stats[(split, typ)] = (H, S)
        print(f"    {cat} {split}/{typ}: {len(n)}장", flush=True)
    del m
    torch.cuda.empty_cache()
    return stats


def deviation(train_a, val_a):
    """val 각 장이 train 클래스면적 분포에서 벗어난 최대 sigma."""
    mu, sd = train_a.mean(0), train_a.std(0)
    return np.max(np.abs(val_a - mu) / (sd + 1.0), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(NUM_CLS))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"{'범주':22s} {'hard 최대σ':>10s} {'soft 최대σ':>10s} "
          f"{'hard >5σ':>9s} {'soft >5σ':>9s}")
    tot_h = tot_s = 0
    for cat in a.cats.split(","):
        st = run_cat(cat, a.seed)
        th, ts = st[("train", "good")]
        vh, vs = st[("validation", "good")]
        dh, ds = deviation(th, vh), deviation(ts, vs)
        nh, ns = int((dh > 5).sum()), int((ds > 5).sum())
        tot_h += nh; tot_s += ns
        print(f"{cat:22s} {dh.max():10.2f} {ds.max():10.2f} {nh:9d} {ns:9d}", flush=True)
    print(f"\n합계 이상치(>5σ): hard {tot_h} -> soft {tot_s}")


if __name__ == "__main__":
    main()
