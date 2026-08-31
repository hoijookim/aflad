#!/usr/bin/env python3
"""PSAD 재구축 2단계 — 구성분기 내부 PatchCore('p' 항) 점수 생성.

psad.py 가 memory_type 과 무관하게 로드하는 per-image .pt 를 만든다:
  {PC_OUT}/{cat}/{split}/{type}/{name}.pt  = {"anomaly_scores": tensor(스칼라)}
  {PC_OUT}/{cat}/ADscore.txt               = 헤더+CSV (psad.py 가 파싱만 하고 미사용)

구현은 공식 patchcore(torch_model.Patchcore, WRN-101-2, layer2+3, 256², k=9,
coreset ratio 1.0=전체 뱅크)를 그대로 사용한다. run.py 의 seed=2 고정 + ratio 1.0
이므로 결정적 — 시드별 재생성이 불필요해 cat 당 1벌이면 된다.

주의: 공식 forward 는 embedding 을 CPU 로 내려 cdist 하므로 뱅크가 커지면 매우
느리다. 여기서는 뱅크·임베딩을 GPU 에 두고 동일 수식(k=9 최근접 + reweight)을
직접 계산한다 — anomaly_map.compute_anomaly_score 와 수치 동일 (blur 는 맵 전용).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PCROOT = Path("/workspace/ai-vision-research/external/PSAD_official/patchcore")
sys.path.insert(0, str(PCROOT))
sys.path.insert(0, "/workspace/ai-vision-research/scripts/psad_rebuild/_shims")
import pl_compat  # noqa: E402,F401  PL 1.x callbacks.base 경로 복원(타입 주석 전용, 수치 불변)
from torch_model import PatchcoreModel  # noqa: E402

DATA = Path("/workspace/ai-vision-research/external/PSAD_official/LOCO_MVTec_AD")
PC_OUT = DATA / "patchcore_score"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SPLITS = [("train", "good"), ("validation", "good"),
          ("test", "good"), ("test", "logical_anomalies"), ("test", "structural_anomalies")]
SIZE = 256
SEED = 2  # 공식 run.py 고정값

tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def embed_image(model, path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).cuda()
    model.training = True                # embedding 반환 모드
    emb = model(x)                 # [n_patch, d]
    return emb


def image_score(emb, bank, k=9, exclude=None):
    """공식 nearest_neighbors + compute_anomaly_score 와 동일 수식 (GPU).

    exclude=(lo,hi): 뱅크의 해당 열 구간을 무한대로 막아 leave-one-out 을 구현한다.
    공식 compute_anomaly_score_standardization(is_training=True) 이 train 점수를 낼 때
    자기 이미지의 패치를 뱅크에서 빼는 것과 수치적으로 동일 —
    이걸 빠뜨리면 train 자기거리가 0 이 되어 정규화 분모가 붕괴한다.
    """
    d = torch.cdist(emb, bank, p=2.0)            # [P, N]
    if exclude is not None:
        d[:, exclude[0]:exclude[1]] = float("inf")
    patch_scores, _ = d.topk(k=k, largest=False, dim=1)  # [P, k]
    patch_scores = patch_scores / 10.0
    if patch_scores.shape[1] != 1:
        max_idx = torch.argmax(patch_scores[:, 0])
        confidence = patch_scores[max_idx].unsqueeze(0)   # [1, k]
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * torch.max(patch_scores[:, 0])
    else:
        score = torch.max(patch_scores[:, 0])
    return score.detach().cpu()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for cat in CATS:
        if only and cat != only:
            continue
        out_root = PC_OUT / cat
        marker = out_root / ".done"
        if marker.exists():
            print(f"[skip] {cat} 완료됨", flush=True)
            continue
        print(f"=== {cat} ===", flush=True)
        model = PatchcoreModel(input_size=(SIZE, SIZE), backbone="wide_resnet101_2",
                               layers=["layer2", "layer3"]).cuda()
        model.feature_extractor.eval()

        # 뱅크: train good 전체 (coreset ratio 1.0 = 서브샘플 없음)
        train_paths = sorted((DATA / "orig_512" / cat / "train" / "good").glob("*.png"))
        feats = []
        for i, p in enumerate(train_paths):
            feats.append(embed_image(model, p))
            if i % 50 == 0:
                print(f"  bank {i}/{len(train_paths)}", flush=True)
        bank = torch.cat(feats, 0)                       # GPU 유지
        print(f"  bank shape={tuple(bank.shape)} "
              f"({bank.numel()*4/1e9:.2f} GB)", flush=True)

        P = feats[0].shape[0]        # 이미지당 패치 수 (256/8)^2 = 1024
        lines = ["name,score"]
        n = 0
        for split, typ in SPLITS:
            src = DATA / "orig_512" / cat / split / typ
            if not src.exists():
                continue
            dst = out_root / split / typ
            dst.mkdir(parents=True, exist_ok=True)
            is_train = (split == "train")
            for i, p in enumerate(sorted(src.glob("*.png"))):
                # train 은 자기 패치 구간을 뱅크에서 제외 (공식 is_training=True 경로)
                exc = (i * P, (i + 1) * P) if is_train else None
                s = image_score(embed_image(model, p), bank, exclude=exc)
                torch.save({"anomaly_scores": s}, dst / f"{p.stem}.pt")
                lines.append(f"{split}/{typ}/{p.stem},{float(s):.6f}")
                n += 1
            print(f"  {split}/{typ}: 완료{' (LOO)' if is_train else ''}", flush=True)
        (out_root / "ADscore.txt").write_text("\n".join(lines) + "\n")
        marker.touch()
        print(f"  [OK] {cat}: {n}개 .pt 저장", flush=True)
        del model, bank, feats
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
