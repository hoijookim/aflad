#!/usr/bin/env python3
"""PSAD 재구축 1단계 — 공식 파이프라인 입력 생성 (d2_csad_to_psad.py 재작성).

생성물 (external/PSAD_official/LOCO_MVTec_AD/ 아래, psad.py 기본 data_path 규약):
  orig_512/{cat}/{split}/{type}/NNN.png   : LOCO 원본 -> 512x512 (RGB bilinear)
      splits = train/good, validation/good, test/{good,logical_anomalies,structural_anomalies}
  {SEG_DIR}/{cat}/pred_NNN.png            : train good 의사레이블 (512x512 uint8 클래스맵)

의사레이블 소스: CSAD 저자 배포 filtered_cluster_map.png (원본 해상도, 부품 클러스터).
  - 원본 해상도 -> 512x512 NEAREST
  - pushpins 특례: CSAD 는 핀을 단일 클래스(1)로 뭉치므로 개수/위치 신호가 히스토그램에서
    "총면적" 하나로 붕괴한다. 원본 d2 정본(num_cls=26, "5x5+bg")을 따라 핀 픽셀을
    5x5 위치 그리드 셀(1..25)로 분해한다.

클래스 수는 공식 psad.py/train_normal_unet.py 하드코딩(7/9/26/7/10) 그대로 두어도 안전:
  각 범주 클래스 합집합(7/6/26/6/3)이 전부 num_cls 미만이라 onehot scatter 가 성립하고,
  빈 클래스는 히스토그램에서 상수 0 열이 될 뿐이다.

검증: 생성 후 범주별 (이미지 수, 클래스 합집합, 512 정합)을 자체 점검한다.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LOCO = Path("/workspace/ai-vision-research/datasets/MVTecLOCO")
import os

# 재생성 마스크와 저자 마스크를 나란히 만들어 비교할 수 있게 경로를 환경변수로 연다.
# 기본값은 기존 동작 그대로다.
MASKS = Path(os.environ.get(
    "CSAD_MASKS", "/workspace/ai-vision-research/external/CSAD_official/datasets/masks"))
OUT = Path("/workspace/ai-vision-research/external/PSAD_official/LOCO_MVTec_AD")
SEG_DIR = os.environ.get("PSAD_SEG_DIR", "csad_pseudo_seg")   # psad.py/train_normal_unet 의 --seg_dir
# E10 splicing 좌/우 분리는 기각된 탐색이다. 기본 끔 — 켜면 보고 수치가 재현되지 않는다.
SPLICING_SPLIT = os.environ.get("PSAD_SPLICING_SPLIT", "0") == "1"
SIZE = 512
ROWS, COLS = 3, 5                     # pushpins 트레이 = 3행 5열 (실측 성분 15개)

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SPLITS = [("train", "good"), ("validation", "good"),
          ("test", "good"), ("test", "logical_anomalies"), ("test", "structural_anomalies")]


def build_orig512(cat: str) -> int:
    n = 0
    for split, typ in SPLITS:
        src = LOCO / cat / split / typ
        if not src.exists():
            continue
        dst = OUT / "orig_512" / cat / split / typ
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.png")):
            o = dst / p.name
            if o.exists():
                n += 1
                continue
            Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(o)
            n += 1
    return n


def pushpins_grid(lab: np.ndarray) -> np.ndarray:
    """핀을 연결요소 단위로 잡아 3x5 컴파트먼트 셀 1..15 로 분해. bg=0.

    데이터 실측: pushpins 는 성분 정확히 15개, y 3밴드 x 5밴드 = 3행 5열 트레이다.
    (공식 psad.py 의 `num_cls = 26  # 16` 주석에서 16 = 15칸 + 배경.)

    픽셀 위치로 나누면 경계에 걸친 핀이 두 셀로 쪼개지므로, 성분 중심(centroid)으로
    셀을 정해 성분 전체에 한 라벨을 준다 — 핀 하나가 사라지면 해당 셀 면적이 0 이 되어
    면적 히스토그램에 직접 신호가 된다.
    """
    from scipy import ndimage
    h, w = lab.shape
    fg = lab > 0
    comp, n = ndimage.label(fg)
    out = np.zeros((h, w), np.uint8)
    if n == 0:
        return out
    sizes = ndimage.sum(fg, comp, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s > 50]     # 잡티 제거
    if not keep:
        return out
    cents = ndimage.center_of_mass(fg, comp, keep)
    for cid, (cy, cx) in zip(keep, cents):
        r = min(int(cy / h * ROWS), ROWS - 1)
        c = min(int(cx / w * COLS), COLS - 1)
        out[comp == cid] = r * COLS + c + 1                  # 1..15
    return out


def screwbag_rank(lab: np.ndarray, k: int = 6) -> np.ndarray:
    """전경 성분을 면적 내림차순 순위로 라벨링 (1..k). bg=0.

    CSAD 의 의미 군집은 screw_bag 에서 이미지마다 클래스 수가 3~5 로 흔들려
    (정상 간 변동계수 0.330) 면적 히스토그램이 불안정하다. 반면 성분을 면적순으로
    정렬한 프로파일은 변동계수 0.059 로 매우 안정적이다 —
    평균 [25997, 19073, 7274, 7109, 6319, 5969] = [봉투, ?, 긴나사x2, 짧은나사x2].
    크기 순서가 곧 부품 정체성이므로 순위 라벨이 의미적으로도 일관된다.
    나사 하나가 빠지면 프로파일이 밀리면서 히스토그램에 직접 신호가 된다.
    """
    from scipy import ndimage
    fg = lab > 0
    comp, n = ndimage.label(fg)
    out = np.zeros(lab.shape, np.uint8)
    if n == 0:
        return out
    sizes = ndimage.sum(fg, comp, range(1, n + 1))
    order = [(s, i + 1) for i, s in enumerate(sizes) if s > 200]
    order.sort(reverse=True)                       # 면적 내림차순
    for rank, (_, cid) in enumerate(order[:k], start=1):
        out[comp == cid] = rank
    return out


def splicing_split_connectors(lab: np.ndarray) -> np.ndarray:
    """splicing_connectors: 클래스 1(커넥터 쌍)을 좌/우로 분리한다. bg=0.

    데이터 실측(train 120장 전수): cls1 은 **모든 장에서 정확히 성분 2개**이고 크기가
    거의 같다(예: 9344 / 9274). CSAD 군집은 시각적으로 동일한 두 커넥터를 같은 "부품
    종류"로 묶는데, 그러면 커넥터별 정보가 **면적 합 하나로 붕괴**한다. LOCO 의 논리
    이상에는 **클램프 개수가 다른 두 커넥터가 짝지어진 경우**가 있고, 합쳐진 표현에서는
    한쪽이 커지고 한쪽이 작아져 합이 거의 그대로 남아 상쇄된다. 클래스별 특징(c 항)도
    두 커넥터의 평균이 되어 희석된다.

    pushpins 에서 핀 15개를 3x5 격자로 분해한 것과 같은 처방이다(그쪽은 원본 대비
    +1.6pp). 여기서는 x 무게중심으로 좌/우를 가른다 — 결정적이고 학습이 없다.
    사용 클래스는 {1,2} 뿐이므로 오른쪽 커넥터에 미사용 번호 3 을 준다(num_cls=10).
    """
    from scipy import ndimage
    out = lab.copy()
    comp, n = ndimage.label(lab == 1)
    if n < 2:
        return out
    sizes = ndimage.sum(lab == 1, comp, range(1, n + 1))
    keep = [i + 1 for i, sz in enumerate(sizes) if sz > 500]
    if len(keep) != 2:
        return out                                  # 2개가 아니면 손대지 않는다
    cx = [float(np.nonzero(comp == c)[1].mean()) for c in keep]
    right = keep[int(np.argmax(cx))]
    out[comp == right] = 3
    return out


def build_pseudo(cat: str) -> tuple[int, list]:
    dst = OUT / SEG_DIR / cat
    dst.mkdir(parents=True, exist_ok=True)
    train_imgs = sorted((LOCO / cat / "train" / "good").glob("*.png"))
    ids_union = set()
    n = 0
    for p in train_imgs:
        stem = p.stem                                       # '000'
        src = MASKS / cat / stem / "filtered_cluster_map.png"
        if not src.exists():
            print(f"    !! {cat}/{stem}: filtered_cluster_map 없음 — 건너뜀", flush=True)
            continue
        lab = np.array(Image.open(src))
        if lab.ndim == 3:
            lab = lab[..., 0]
        if cat == "pushpins":
            lab = pushpins_grid(lab)
        elif cat == "splicing_connectors" and SPLICING_SPLIT:
            # E10(좌/우 분리)은 **기각됐다** — val 정상 점수 꼬리 max|z_robust| 6.29 -> 54.60
            # (PREREG_branch_improvement.md:545). 보고 수치는 분리 없는 레이블을 쓴다.
            # 기본값을 켜 두면 재생성이 기각된 변형을 조용히 만들어 재현이 깨진다.
            # E10 을 재현하려면 PSAD_SPLICING_SPLIT=1 로 명시적으로 켠다.
            lab = splicing_split_connectors(lab)
        # screw_bag: 순위/면적구간 라벨을 시도했으나 성분 수 변동(4~7)으로
        # 변동계수가 오히려 악화(0.406/0.327 vs CSAD 0.330) — CSAD 군집 유지.
        lab = np.array(Image.fromarray(lab.astype(np.uint8))
                       .resize((SIZE, SIZE), Image.NEAREST))
        ids_union |= set(np.unique(lab).tolist())
        Image.fromarray(lab).save(dst / f"pred_{p.name}")
        n += 1
    return n, sorted(ids_union)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    num_cls = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
               "screw_bag": 7, "splicing_connectors": 10}
    print(f"출력 루트: {OUT}")
    ok = True
    for cat in CATS:
        if only and cat != only:
            continue
        n_img = build_orig512(cat)
        n_lab, ids = build_pseudo(cat)
        fits = max(ids) < num_cls[cat]
        ok &= fits
        print(f"  {cat:22s} orig512 {n_img:4d}장 | pred {n_lab:3d}장 | "
              f"클래스 {ids[:6]}{'...' if len(ids) > 6 else ''} max={max(ids)} "
              f"< num_cls {num_cls[cat]} : {'OK' if fits else 'FAIL'}", flush=True)
    print("검증:", "PASS" if ok else "FAIL — 클래스 범위 초과")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
