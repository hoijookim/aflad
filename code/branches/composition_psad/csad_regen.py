#!/usr/bin/env python3
"""CSAD 의사레이블을 이 환경에서 재생성 — 저자 마스크는 건드리지 않는다.

배경: PSAD 구성 분기는 CSAD 의사분할(filtered_cluster_map.png)을 입력으로 받는다.
지금까지는 저자가 배포한 마스크를 그대로 썼다. 재생성의 목적은 두 가지다.
  (1) 재현성 — 같은 코드가 이 환경에서 같은 품질의 마스크를 내는가
  (2) 개선 여지 — 프롬프트/임계값/군집 설정을 train 전용 기준으로 튜닝할 수 있는가

경로 분리: 원 코드는 mask_path 를 project_root/datasets/masks 로 고정한다. 저자 마스크를
덮어쓰지 않도록 그 이름을 심볼릭 링크로 바꿔 재생성 산출물을 masks_regen 아래로 보낸다.
저자 원본은 masks_author(백업)와 masks_orig(현물) 두 벌로 남는다.

test 는 어느 단계에서도 쓰지 않는다. 입력은 train/good 이미지뿐이다.
"""
import argparse
import glob
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

CSAD = Path("/workspace/ai-vision-research/external/CSAD_official")
os.chdir(CSAD)
sys.path.insert(0, str(CSAD))

MASKS = CSAD / "datasets/masks"
ORIG = CSAD / "datasets/masks_orig"
REGEN = CSAD / "datasets/masks_regen"


def swap_in():
    """datasets/masks -> masks_regen 심볼릭 링크. 저자 원본은 masks_orig 로 이동."""
    if MASKS.is_symlink():
        return
    if not ORIG.exists():
        shutil.move(str(MASKS), str(ORIG))
    elif MASKS.exists():
        shutil.rmtree(MASKS)
    REGEN.mkdir(parents=True, exist_ok=True)
    MASKS.symlink_to(REGEN.name)


def swap_out():
    """저자 마스크를 원위치로 되돌린다."""
    if MASKS.is_symlink():
        MASKS.unlink()
    if ORIG.exists() and not MASKS.exists():
        shutil.move(str(ORIG), str(MASKS))


def run(category):
    import timm
    import torch
    from torchvision import transforms
    import pseudo_label as PL

    # 원 코드는 `if __name__` 블록에서 timm/transforms 를 모듈 전역에 주입한 뒤
    # ClassHistogram 을 만든다. 그 블록을 우회하므로 여기서 같은 주입을 해준다.
    PL.transforms = transforms
    PL.timm = timm

    cfg = PL.read_config(f"{CSAD}/configs/class_histogram/{category}.yaml")
    cfg["project_root"] = str(CSAD)
    cfg["mask_path"] = str(MASKS)
    cfg["ckpt_path"] = str(CSAD / "ckpt")
    cfg["grounding_config"]["project_root"] = cfg["project_root"]
    cfg["grounding_config"]["ckpt_path"] = cfg["ckpt_path"]
    # 저자 코드의 segmentation() 은 인자로 받지 않은 전역 `config` 를 참조한다
    # (원 __main__ 이 루프 변수로 모듈 전역에 심어두기 때문에 거기서만 동작한다).
    PL.config = cfg

    paths = sorted(glob.glob(
        f"{CSAD}/datasets/mvtec_loco_anomaly_detection/{cfg['category']}/train/good/*.png"))
    print(f"[{category}] train/good {len(paths)}장", flush=True)

    # 마스크 생성은 단계별로 재개 가능하게 직접 호출한다. ClassHistogram.__init__ 은
    # 범주 디렉터리가 이미 있으면 두 단계를 통째로 건너뛰므로, 중간에 끊기면
    # grounding 만 있고 refined 가 없는 상태로 남는다.
    md = Path(cfg["mask_path"]) / category
    md.mkdir(parents=True, exist_ok=True)
    done_g = len(list(md.glob("*/grounding_mask.png")))
    if done_g < len(paths):
        print(f"  [{category}] grounding ({done_g}/{len(paths)} 완료됨)", flush=True)
        PL.grounding_segmentation(paths, str(md), cfg["grounding_config"])
    done_s = len(list(md.glob("*/refined_masks.png")))
    if done_s < len(paths):
        print(f"  [{category}] SAM 분할 ({done_s}/{len(paths)} 완료됨)", flush=True)
        PL.segmentation(paths, str(md),
                        use_grounding_filter=cfg["use_grounding_filter"],
                        no_sam=cfg["no_sam"])

    fe = timm.create_model("wide_resnet50_2", pretrained=True,
                           features_only=True, out_indices=[4]).eval().cuda()
    ch = PL.ClassHistogram(paths, feature_extractor=fe, config=cfg)
    # __init__ 은 마스크 생성과 성분 특징까지만 한다. 군집 -> 지도 저장 -> 히스토그램 순.
    for step in ("cluster_component_features", "save_cluster_map", "build_histogram",
                 "remap_cluster_map_and_histogram"):
        fn = getattr(ch, step, None)
        if fn is None:
            continue
        print(f"  [{category}] {step}", flush=True)
        fn()
    del fe, ch
    torch.cuda.empty_cache()

    n = len(glob.glob(f"{REGEN}/{category}/*/filtered_cluster_map.png"))
    print(f"[{category}] 완료 — filtered_cluster_map {n}개", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="screw_bag")
    ap.add_argument("--restore", action="store_true", help="심볼릭 링크 해제 후 종료")
    a = ap.parse_args()
    if a.restore:
        swap_out()
        print("저자 마스크 원위치 복구 완료")
        return
    swap_in()
    print(f"산출 경로: {REGEN} (저자 원본은 {ORIG})", flush=True)
    for c in a.cats.split(","):
        try:
            run(c)
        except Exception as e:
            print(f"[{c}] 실패: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
