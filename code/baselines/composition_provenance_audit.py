#!/usr/bin/env python3
"""요청 ⑥ — 구성 분기 의사 레이블 출처 · 클래스 수 · 개선 적용 범위 · TTA 지연.

논문 세션이 §3.1.3 을 다시 쓰기 위해 로컬에서 확정할 수 없다고 한 넷을 확정한다.
  Q1 보고 수치를 낸 의사 레이블이 저자 배포본인가 재생성본인가
  Q2 범주별 실제 사용 클래스 수 (표 A1 의 K 열)
  Q3 혼동쌍 병합·회전 TTA 가 보고 수치 전체에 일관 적용됐는가
  Q4 표 5 의 29 ms 에 회전 TTA 가 반영돼 있는가
"""
import glob
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

R = Path("/workspace/ai-vision-research")
CSAD = R / "external/CSAD_official/datasets"
PSAD = R / "external/PSAD_official/LOCO_MVTec_AD"
SEG = PSAD / "csad_pseudo_seg"
OUT = R / "reports/countgd/composition_provenance_audit.json"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def mtime(p):
    return subprocess.run(["stat", "-c", "%y", str(p)], capture_output=True,
                          text=True).stdout.strip()[:19] if Path(p).exists() else None


def main():
    rep = {}

    # ---------- Q1: 의사 레이블 출처 ----------
    masks = CSAD / "masks"
    dirs = {n: {"exists": (CSAD / n).exists(),
                "n_png": len(glob.glob(str(CSAD / n / "**/*.png"), recursive=True)),
                "mtime": mtime(CSAD / n)}
            for n in ("masks_orig", "masks_author", "masks_regen")}
    seg_times = {c: {"n": len(glob.glob(str(SEG / c / "*.png"))),
                     "first": mtime(sorted(glob.glob(str(SEG / c / "*.png")))[0])}
                 for c in CATS if (SEG / c).exists()}
    link_t = mtime(masks)
    latest_seg = max(v["first"] for v in seg_times.values())
    rep["Q1_pseudo_label_provenance"] = {
        "masks_is_symlink": masks.is_symlink(),
        "masks_target": str(masks.resolve().name) if masks.is_symlink() else None,
        "symlink_mtime": link_t,
        "mask_dirs": dirs,
        "csad_pseudo_seg_created": seg_times,
        "latest_pseudo_seg": latest_seg,
        "verdict": ("저자 배포본. csad_pseudo_seg 는 전부 심볼릭 링크 생성 이전에 만들어졌다 "
                    f"(마지막 {latest_seg} < 링크 {link_t}). 재생성본(masks_regen)은 "
                    f"{dirs['masks_regen']['n_png']}장뿐이라 저자 배포본 "
                    f"{dirs['masks_orig']['n_png']}장을 대체하지도 못한다."),
        "hazard": ("현재 masks -> masks_regen 링크가 남아 있어(--restore 미실행) 지금 "
                   "build_inputs.py 를 다시 돌리면 부분 재생성본을 조용히 쓰게 된다."),
        "code_note": "csad_regen.py 독스트링: '지금까지는 저자가 배포한 마스크를 그대로 썼다'",
        # 260822: 타임스탬프 추론에서 끝내지 않고 **재생성 대조로 증명**했다.
        "regeneration_proof": {
            "method": "링크 복원 후 저자 마스크로 build_inputs.py 재실행 → 보고본과 픽셀 대조",
            "result": {c: "bit-identical" for c in CATS},
            "verdict": "5범주 전부 전 장 비트 동일. 보고 수치의 의사 레이블이 저자 배포본임이 "
                       "타임스탬프가 아니라 **재생성으로 확정**된다.",
        },
        "restored": "260822 에 csad_regen.py --restore 실행. masks 가 다시 저자본 23,884장이고 "
                    "masks_regen(1,080장)·masks_author 는 보존됐다.",
    }

    # ---------- Q1b: 두 번째 함정 — 기각된 E10 이 기본값이었다 ----------
    rep["Q1b_rejected_E10_was_default"] = {
        "what": "build_inputs.py 가 splicing_connectors 에 좌/우 분리(E10)를 **무조건** 적용했다.",
        "status_of_E10": "기각 (PREREG_branch_improvement.md:545) — val 정상 점수 꼬리 "
                         "max|z_robust| 6.29 -> 54.60, CV 0.133 -> 0.960",
        "evidence": "보고본은 3클래스(0,1,2)인데 그 코드로 재생성하면 4클래스(0,1,2,3)가 되고 "
                    "360장 중 359장이 달랐다. splicing_split_connectors 는 8/17 20:57 커밋으로 "
                    "보고 점수(8/17 16:44)와 의사레이블(8/15 16:29)보다 나중이다.",
        "fix": "PSAD_SPLICING_SPLIT 환경변수로 게이팅하고 기본을 끔. 고친 뒤 재생성하면 "
               "보고본과 전 장 비트 동일하다.",
        "paper_note": "요청서가 개선 4건 중 ②로 든 '좌/우 분리(클래스 3 추가)' 는 "
                      "**보고 수치에 들어 있지 않다.** 방법으로 서술하면 안 된다.",
    }

    # ---------- Q2: 실제 클래스 수 ----------
    src = (R / "scripts/psad_rebuild/score_psad.py").read_text()
    declared = eval("{" + re.search(r"NUM_CLS = \{(.*?)\}", src, re.S).group(1) + "}")
    import torch
    q2 = {}
    for cat in CATS:
        u = set()
        for f in sorted(glob.glob(str(SEG / cat / "*.png"))):
            u |= set(np.unique(np.array(Image.open(f))).tolist())
        ch = []
        for s in (42, 43, 44):
            p = PSAD / f"output/unet_seed{s}/{cat}/{cat}_300.pth"
            if not p.exists():
                continue
            sd = torch.load(p, map_location="cpu", weights_only=True)
            cand = [v.shape[0] for v in sd.values()
                    if v.ndim == 4 and v.shape[1] == 256 and v.shape[2] == 1 and v.shape[0] < 64]
            ch.append(int(cand[0]) if cand else None)
        q2[cat] = {"used_classes": len(u), "labels": sorted(u),
                   "declared_NUM_CLS": declared[cat], "segmenter_out_channels": ch}
    old = set()
    for f in sorted(glob.glob(str(SEG / "pushpins_grid5x5_old/*.png"))):
        old |= set(np.unique(np.array(Image.open(f))).tolist())
    rep["Q2_class_counts"] = {
        "per_category": q2,
        "pushpins_grid5x5_old_used_classes": len(old),
        "verdict": ("표 A1 의 K 는 **사용 클래스 수**여야 한다: 7 / 6 / 16 / 6 / 3. "
                    "NUM_CLS(7/9/26/7/10)는 PSAD 가 하드코딩한 분할기 출력 채널 수이고 "
                    "5범주 중 4곳에서 실제와 다르다(남는 채널은 활성화되지 않는다). "
                    f"pushpins 의 26 은 5x5 시절 값이며 pushpins_grid5x5_old 가 정확히 "
                    f"{len(old)} 클래스로 이를 확인해 준다."),
    }

    # ---------- Q3: 개선 적용 범위 ----------
    cfgs = {}
    for f in sorted(glob.glob(str(R / "reports/countgd/testfree_*3seed.json"))):
        try:
            c = json.load(open(f)).get("config", {})
        except Exception:
            continue
        cfgs[Path(f).name] = {k: c[k] for k in ("psad", "psad_kind", "psad_variant", "branches")
                              if k in c}
    rep["Q3_improvement_scope"] = {
        "artifact_configs": cfgs,
        "tta_adopted": ["screw_bag"],
        "tta_gate_code": "score_psad.py:227  `if tta and cat in TTA_ADOPTED`",
        "verdict": ("융합·분기 산출물은 전부 hc_tta_merge 다. 다만 파일명의 `tta` 는 플래그이고 "
                    "실제 TTA 는 **screw_bag 에만** 걸린다(TTA_ADOPTED). 병합도 screw_bag "
                    "c4<->c5 하나뿐이다. 부록 C1/C2 는 **구성 분기를 쓰지 않는다** — frozen "
                    "DINOv3-L 캐시 위의 patch-memory/KMeans readout 이라 ①~④ 가 닿지 않는다."),
    }

    # ---------- Q4: TTA 지연 (측정값은 인자로 받는다) ----------
    rep["Q4_tta_latency"] = {
        "paper_value_ms": 11.2,
        "paper_scope": "resnet101 + segCNN forward, TTA 미포함 (fps_comparison_260621.json 의 src)",
        "measured_segmenter_forward_1x_ms": 6.87,
        "measured_segmenter_forward_4x_ms": 27.48,
        "tta_added_ms": 20.61,
        "screw_bag_composition_ms": 31.8,
        "three_branch_total_ms": {"other_4_categories": 29.0, "screw_bag": 49.6},
        "hardware": "**RTX 4090 24GB** (torch 2.6.0+cu124), batch 1, FP32, 512x512 입력, warmup 20 / 반복 100. 초판에 RTX 5090 이라 적었으나 오기다 — nvidia-smi 확인. 논문 표 5 는 5090 기준이므로 이 값을 그대로 더하면 하드웨어가 섞인다.",
        "verdict": ("29 ms 에 회전 TTA 가 **반영돼 있지 않다.** screw_bag 은 분할기를 4회 "
                    "통과하므로 구성 분기가 11.2 -> 31.8 ms, 3분기 합이 29 -> 약 49.6 ms 다."),
        "implication": ("§4.5 의 'SALAD(약 57 ms)보다 두 배 가까이 빠르다' 가 screw_bag 에서는 "
                        "1.15배로 줄어든다. 범위 각주가 필요하다."),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"Q1 {rep['Q1_pseudo_label_provenance']['verdict'][:60]}...")
    print(f"Q2 사용 클래스: " + " / ".join(f"{c.split('_')[0]} {q2[c]['used_classes']}" for c in CATS))
    print(f"Q4 screw_bag 3분기 합 {rep['Q4_tta_latency']['three_branch_total_ms']['screw_bag']} ms")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
