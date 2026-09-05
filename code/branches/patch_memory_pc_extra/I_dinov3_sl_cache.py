"""Direction I extended: DINOv3-S and DINOv3-L cache extraction.

DINOv3-S: 21.6M, 12 layers, hidden 384 (smaller, faster)
DINOv3-L: 303.1M, 24 layers, hidden 1024 (larger, more capacity)

For S: layers L3/L6/L9/L11 (same as B)
For L: layers L8/L17/L23 (proportional)
"""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

# HF_TOKEN은 환경변수 또는 `huggingface-cli login`으로 설정 (gated model 접근용)

REPO = Path("/workspace/ai-vision-research")
LOCO_DIR = REPO / "datasets" / "MVTecLOCO"

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
RESOLUTION = 336

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Variant config from CLI
VARIANTS = {
    "S": {
        "model": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "cache_dir": REPO / "cache" / "dinov3_multilayer_vits16",
        "layers": [3, 6, 9, 11],
    },
    "L": {
        "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "cache_dir": REPO / "cache" / "dinov3_multilayer_vitl16",
        "layers": [8, 17, 23],
    },
}


def resolve_blocks(m):
    """트랜스포머 블록 ModuleList 를 transformers 버전 무관하게 찾는다.

    transformers <=5.0 : DINOv3ViTModel.layer
    transformers 5.15  : DINOv3ViTModel.model.layer (중첩 한 단계 추가)
    블록 자체는 동일하므로 추출 특징은 두 경로에서 수치적으로 같다.
    """
    for path in ("layer", "model.layer", "encoder.layer"):
        obj = m
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return obj
    raise RuntimeError(
        f"{type(m).__name__}: 블록 ModuleList 경로를 찾지 못함 "
        f"(children={[n for n, _ in m.named_children()]})")


def collect_split(cat, split):
    items = []
    base = LOCO_DIR / cat / split
    if not base.exists():
        return items
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        lt = sub.name
        if lt == "good":
            lab = 0
            lt_short = "good"
        elif "logical" in lt:
            lab = 1
            lt_short = "logical"
        elif "structural" in lt:
            lab = 1
            lt_short = "structural"
        else:
            continue
        for p in sorted(sub.glob("*.png")):
            items.append((p, lab, lt_short))
    return items


def cache_variant(variant):
    spec = VARIANTS[variant]
    OUT_DIR = spec["cache_dir"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYERS = spec["layers"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== DINOv3-{variant} cache extraction ===")
    print(f"Model: {spec['model']}, Layers: {LAYERS}")

    print(f"Loading...")
    model = AutoModel.from_pretrained(spec["model"], torch_dtype=torch.float32)
    model = model.to(device).eval()
    print(f"  loaded, n_params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    blocks = resolve_blocks(model)
    n_blocks = len(blocks)
    print(f"  n_blocks: {n_blocks}")

    layer_outputs = {}
    def make_hook(layer_idx):
        def hook(module, inp, out):
            layer_outputs[layer_idx] = out[0] if isinstance(out, tuple) else out
        return hook
    for L in LAYERS:
        blocks[L].register_forward_hook(make_hook(L))

    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    for cat in CATS:
        out_path = OUT_DIR / f"{cat}.npz"
        if out_path.exists():
            print(f"[skip] {cat} already cached")
            continue
        print(f"\n  === {cat} ===")
        train_good = [item for item in collect_split(cat, "train") if item[2] == "good"]
        val_good = [item for item in collect_split(cat, "validation") if item[2] == "good"]
        test = collect_split(cat, "test")
        print(f"    train_good={len(train_good)} val_good={len(val_good)} test={len(test)}")

        all_features = {L: {"train": [], "test": []} for L in LAYERS}
        all_labels_test = []
        all_ltypes_test = []

        for split_name, items in [("train", train_good + val_good), ("test", test)]:
            for i, (path, lab, lt) in enumerate(items):
                if i % 50 == 0:
                    print(f"      {split_name} {i}/{len(items)}", flush=True)
                try:
                    image = Image.open(path).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        layer_outputs.clear()
                        _ = model(tensor)
                    for L in LAYERS:
                        h = layer_outputs[L]
                        feat = h[0, 5:, :].cpu().numpy().astype(np.float32)  # drop CLS + 4 register
                        all_features[L][split_name].append(feat)
                    if split_name == "test":
                        all_labels_test.append(lab)
                        all_ltypes_test.append(lt)
                except Exception as e:
                    print(f"      [ERR] {path}: {e}")

        save_dict = {}
        for L in LAYERS:
            save_dict[f"train_L{L}"] = np.stack(all_features[L]["train"])
            save_dict[f"test_L{L}"] = np.stack(all_features[L]["test"])
        save_dict["test_labels"] = np.array(all_labels_test)
        save_dict["test_ltypes"] = np.array(all_ltypes_test)

        np.savez_compressed(out_path, **save_dict)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    saved {out_path}  size={size_mb:.1f}MB")
        for L in LAYERS:
            print(f"      L{L} train: {save_dict[f'train_L{L}'].shape}, test: {save_dict[f'test_L{L}'].shape}")

    print(f"\n[OK] DINOv3-{variant} cache complete: {OUT_DIR}")
    # Free GPU
    del model
    torch.cuda.empty_cache()


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "S"
    if variant not in VARIANTS:
        print(f"Unknown variant {variant}. Choose from {list(VARIANTS.keys())}")
        return
    cache_variant(variant)


if __name__ == "__main__":
    main()
