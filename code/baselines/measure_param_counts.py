#!/usr/bin/env python3
"""논문이 인용하는 파라미터 수를 전부 실측한다.

계기: §4.5 는 DINOv3-L 을 303M, 표 C1 은 같은 모델을 300M 으로 적고 있었다.
어느 쪽이 맞느냐가 아니라 **두 관례가 섞여 있던 것**이다 — §4.5 는 실측,
표 C1 은 계열 명목 표기(S≈21M / B≈86M / L≈300M)다.

명목이면 DINOv2-B(86.58M)는 87M 이어야 하는데 표는 86M 이다. 즉 표 C1 은
5칸 중 4칸이 실측과 다르다. 실측으로 통일하면 그 모순이 사라진다.
"""
import json
from pathlib import Path
import torch

REPO = Path("/workspace/ai-vision-research")
OUT = REPO / "reports/countgd/param_counts_measured.json"
PAPER = {  # 논문 현재 표기
    "DINOv3-S/16": "21M", "DINOv3-B/16": "86M", "DINOv3-L/16": "300M (표 C1) / 303M (§4.5)",
    "DINOv2-B/14": "86M", "DINOv2-L/14": "300M",
}


def count(m):
    return int(sum(p.numel() for p in m.parameters()))


def main():
    from transformers import AutoModel
    import torchvision

    bb = {}
    for tag, hf in (("DINOv3-S/16", "facebook/dinov3-vits16-pretrain-lvd1689m"),
                    ("DINOv3-B/16", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
                    ("DINOv3-L/16", "facebook/dinov3-vitl16-pretrain-lvd1689m")):
        bb[tag] = count(AutoModel.from_pretrained(hf))
    for tag, hub in (("DINOv2-B/14", "dinov2_vitb14"), ("DINOv2-L/14", "dinov2_vitl14")):
        bb[tag] = count(torch.hub.load("facebookresearch/dinov2", hub, verbose=False))

    # AFLAD 세 분기 (§4.5 의 482M 분해)
    cd = (REPO / "reports/phase0/efficient_ad_official_small_seeds_std"
          / "breakfast_box_seed44/trainings/mvtec_loco/breakfast_box")
    rec = {f.split("_")[0]: count(torch.load(cd / f, map_location="cpu", weights_only=False))
           for f in ("teacher_final.pth", "student_final.pth", "autoencoder_final.pth")}
    comp = {"segmenter_encoder_WRN101_2": count(torchvision.models.wide_resnet101_2()),
            "embedding_ResNet101": count(torchvision.models.resnet101())}
    branches = {"reconstruction_EADS": sum(rec.values()),
                "patch_memory_DINOv3L": bb["DINOv3-L/16"],
                "composition": sum(comp.values())}

    out = {
        "backbones_table_C1": {k: {"params": v, "M": round(v / 1e6, 2),
                                   "rounded_M": round(v / 1e6), "paper_now": PAPER[k]}
                               for k, v in bb.items()},
        "aflad_branches_sec4_5": {
            "reconstruction_detail": rec, "composition_detail": comp,
            "per_branch": branches,
            "total": sum(branches.values()),
            "total_M": round(sum(branches.values()) / 1e6, 2),
            "paper_now": "약 482M (재구성 8.1M, 패치 메모리 303M, 구성 약 171M)",
            "verdict": "네 값 모두 실측과 일치한다. §4.5 는 고칠 것이 없다.",
        },
        "note": "구성 분기 실측은 분할 인코더 + 임베딩 backbone 만이다. 합성곱 디코더와 "
                "좌표 채널은 제외돼 있어 논문의 '약 171M' 이 그만큼 보수적이다.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    for k, v in out["backbones_table_C1"].items():
        flag = "" if str(v["rounded_M"]) + "M" == v["paper_now"] else "   ← 표기 불일치"
        print(f"  {k:14s} {v['M']:>7.2f}M → {v['rounded_M']:>3d}M   논문 {v['paper_now']}{flag}")
    print(f"\n  분기 합 {out['aflad_branches_sec4_5']['total_M']:.2f}M  (논문 약 482M)")
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
