#!/usr/bin/env python3
"""EAD-S 임의 시드 추론 → score npz (ead_infer_seeds_43_44.py 의 시드 일반형).

원본과의 차이는 두 가지뿐이다.
  1. ckpt_root 를 항상 `_seeds_std/{cat}_seed{S}` 로 고정 (원본은 42만 다른 경로를 봤다 —
     그 경로의 체크포인트는 이 워크스페이스에 없다).
  2. --compare 로 기존 정본 npz 와 LS AUROC / 상관을 대조한다.
추론 로직(정렬 규약·score 정의)은 원본과 동일하게 유지했다:
  DEFECTS=[good,logical_anomalies,structural_anomalies] × sorted glob,
  score = max(pad(map,4) 를 원본 해상도로 bilinear 보간한 값).

## 260822 추가 — RNG 고정 (--rng, 기본 0)

추론 경로에 난수가 들어간다. `E.train_transform` 이
`RandomChoice([ColorJitter(brightness), ColorJitter(contrast), ColorJitter(saturation)])`
를 적용하는데, 이 변환이 **정규화 상수 계산에 쓰인다** —
`teacher_normalization`(tmean/tstd) 과 `map_normalization`(분위수) 둘 다.
train_loader 의 `shuffle=True` 도 부동소수점 누적 순서를 바꾼다.
공식 `efficientad.py` main 은 `torch.manual_seed/np.random.seed/random.seed` 를 걸지만
이 추론 스크립트에는 없었다 — 그래서 같은 체크포인트로도 실행마다 값이 달라질 수 있었다.

기본값 0 으로 고정한다. **주의: 기존 정본 npz 는 이 인자가 없던 시절 산출물이므로,
이 스크립트로 그 값을 그대로 되살릴 수 있다고 가정하지 말 것.** 재현 여부는 --compare 로
확인한다. (5090 검증: `_seeds_std` 6월 체크포인트 + --rng 0 은 6월 npz 를 1e-6 이내로
재현했다. 8월 정본과의 차이는 RNG 가 아니라 체크포인트 계보 차이였다.)

사용:
  python3 scripts/phase0/ead_infer_seeds.py --seeds 42 --compare   # 충실도 검증
  python3 scripts/phase0/ead_infer_seeds.py --seeds 43,44,45,46    # 본 산출
  python3 scripts/phase0/ead_infer_seeds.py --seeds 43 --rng 1     # 추론 잡음 측정
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

EADDIR = "/workspace/ai-vision-research/external/efficient_ad_official"
sys.path.insert(0, EADDIR)
import efficientad as E  # noqa: E402
from common import ImageFolderWithoutTarget  # noqa: E402

R = Path("/workspace/ai-vision-research")
LOCO = R / "datasets/MVTecLOCO"
NPZOUT = R / "reports/phase0/efficient_ad_official_small/npz"
CKPT = R / "reports/phase0/efficient_ad_official_small_seeds_std"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
DEFECTS = ["good", "logical_anomalies", "structural_anomalies"]
DLT = {"good": "good", "logical_anomalies": "logical", "structural_anomalies": "structural"}


@torch.no_grad()
def infer(cat, seed):
    tr = CKPT / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat
    teacher = torch.load(tr / "teacher_final.pth", map_location="cpu", weights_only=False).eval()
    student = torch.load(tr / "student_final.pth", map_location="cpu", weights_only=False).eval()
    autoenc = torch.load(tr / "autoencoder_final.pth", map_location="cpu", weights_only=False).eval()
    if E.on_gpu:
        teacher.cuda(); student.cuda(); autoenc.cuda()

    lam = torchvision.transforms.Lambda(E.train_transform)
    train_loader = DataLoader(ImageFolderWithoutTarget(str(LOCO / cat / "train"), transform=lam),
                              batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ImageFolderWithoutTarget(str(LOCO / cat / "validation"), transform=lam),
                            batch_size=1)
    tmean, tstd = E.teacher_normalization(teacher, train_loader)
    qss, qse, qas, qae = E.map_normalization(val_loader, teacher, student, autoenc, tmean, tstd)

    scores, labels, ltypes = [], [], []
    for d in DEFECTS:
        dd = LOCO / cat / "test" / d
        if not dd.exists():
            continue
        for p in sorted(dd.glob("*.png")):
            img = Image.open(p).convert("RGB")
            ow, oh = img.width, img.height
            x = E.default_transform(img)[None]
            if E.on_gpu:
                x = x.cuda()
            mc, _, _ = E.predict(image=x, teacher=teacher, student=student, autoencoder=autoenc,
                                 teacher_mean=tmean, teacher_std=tstd,
                                 q_st_start=qss, q_st_end=qse, q_ae_start=qas, q_ae_end=qae)
            mc = torch.nn.functional.pad(mc, (4, 4, 4, 4))
            mc = torch.nn.functional.interpolate(mc, (oh, ow), mode="bilinear")[0, 0].cpu().numpy()
            scores.append(float(np.max(mc)))
            labels.append(0 if d == "good" else 1)
            ltypes.append(DLT[d])
    return np.array(scores, np.float64), np.array(labels, np.int64), np.array(ltypes)


def ls_auc(s, lab, lt):
    return 0.5 * sum(roc_auc_score(lab[(lt == "good") | (lt == k)], s[(lt == "good") | (lt == k)])
                     for k in ["logical", "structural"])


@torch.no_grad()
def infer_val(cat, seed):
    """validation/good 점수 (test-free z-norm 통계용). infer() 와 동일 정규화 경로."""
    tr = CKPT / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat
    teacher = torch.load(tr / "teacher_final.pth", map_location="cpu", weights_only=False).eval()
    student = torch.load(tr / "student_final.pth", map_location="cpu", weights_only=False).eval()
    autoenc = torch.load(tr / "autoencoder_final.pth", map_location="cpu", weights_only=False).eval()
    if E.on_gpu:
        teacher.cuda(); student.cuda(); autoenc.cuda()
    lam = torchvision.transforms.Lambda(E.train_transform)
    train_loader = DataLoader(ImageFolderWithoutTarget(str(LOCO / cat / "train"), transform=lam),
                              batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ImageFolderWithoutTarget(str(LOCO / cat / "validation"), transform=lam),
                            batch_size=1)
    tmean, tstd = E.teacher_normalization(teacher, train_loader)
    qss, qse, qas, qae = E.map_normalization(val_loader, teacher, student, autoenc, tmean, tstd)
    scores, fnames = [], []
    for p2 in sorted((LOCO / cat / "validation" / "good").glob("*.png")):
        img = Image.open(p2).convert("RGB")
        ow, oh = img.width, img.height
        x = E.default_transform(img)[None]
        if E.on_gpu:
            x = x.cuda()
        mc, _, _ = E.predict(image=x, teacher=teacher, student=student, autoencoder=autoenc,
                             teacher_mean=tmean, teacher_std=tstd,
                             q_st_start=qss, q_st_end=qse, q_ae_start=qas, q_ae_end=qae)
        mc = torch.nn.functional.pad(mc, (4, 4, 4, 4))
        mc = torch.nn.functional.interpolate(mc, (oh, ow), mode="bilinear")[0, 0].cpu().numpy()
        scores.append(float(np.max(mc)))
        fnames.append(p2.name)
    return np.array(scores, np.float64), np.array(fnames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="43,44,45,46")
    ap.add_argument("--compare", action="store_true",
                    help="기존 정본 npz 와 대조만 하고 저장하지 않는다")
    ap.add_argument("--val", action="store_true",
                    help="validation/good 점수 추출 -> val_good_scores npz 저장")
    ap.add_argument("--rng", type=int, default=0,
                    help="난수 고정값. 정규화 상수가 랜덤 증강에 의존하므로 필요하다. "
                         "다른 값으로 돌리면 추론 잡음의 크기를 잴 수 있다.")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    def fix_rng():
        """매 (범주, 시드) 조합 앞에서 같은 상태로 되돌린다 — 순서에 무관하게 재현되도록."""
        torch.manual_seed(a.rng)
        np.random.seed(a.rng)
        random.seed(a.rng)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.rng)

    if a.val:
        outdir = R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
        outdir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for cat in CATS:
                ck = CKPT / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat / "teacher_final.pth"
                if not ck.exists():
                    print(f"  [skip] {cat}/seed{seed}", flush=True)
                    continue
                if (outdir / f"val_good_{cat}_seed{seed}.npz").exists():
                    print(f"  [skip-done] {cat}/seed{seed}", flush=True)
                    continue
                fix_rng()
                sc, fn = infer_val(cat, seed)
                np.savez(outdir / f"val_good_{cat}_seed{seed}.npz", scores=sc, fnames=fn)
                print(f"  [OK] val {cat}/seed{seed} N={len(sc)} mean={sc.mean():.4f}", flush=True)
        print("DONE", flush=True)
        return

    cross = {}
    for seed in seeds:
        lss = []
        for cat in CATS:
            ck = CKPT / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat / "teacher_final.pth"
            if not ck.exists():
                print(f"  [skip] {cat}/seed{seed}: 체크포인트 없음", flush=True)
                continue
            fix_rng()
            s, lab, lt = infer(cat, seed)
            new_ls = ls_auc(s, lab, lt)
            lss.append(new_ls)
            if a.compare:
                ref = NPZOUT / f"scores_{cat}_seed{seed}.npz"
                if ref.exists():
                    old = np.load(ref)
                    old_ls = ls_auc(old["score"], old["label"], old["label_type"])
                    same_n = len(s) == len(old["score"])
                    corr = float(np.corrcoef(s, old["score"])[0, 1]) if same_n else float("nan")
                    print(f"  [{cat:22s}] n={len(s)}(정본 {len(old['score'])}) "
                          f"LS 재현={new_ls:.4f} 정본={old_ls:.4f} Δ={abs(new_ls-old_ls):.4f} "
                          f"corr={corr:.4f}", flush=True)
                    cross[cat] = {"repro": float(new_ls), "canonical": float(old_ls),
                                  "corr": corr, "n": int(len(s)), "seed": seed}
                else:
                    print(f"  [{cat:22s}] LS={new_ls:.4f} (정본 npz 없음 — 대조 불가)", flush=True)
            else:
                np.savez(NPZOUT / f"scores_{cat}_seed{seed}.npz",
                         score=s, label=lab, label_type=lt)
                print(f"  [OK] {cat}/seed{seed} N={len(s)} LS={new_ls:.4f}", flush=True)
        if lss:
            print(f"== seed {seed}: 5-cat mean LS = {np.mean(lss):.4f}\n", flush=True)
    if cross:
        import json
        out = R / "_logs/ead_seed42_crosscheck.json"
        out.parent.mkdir(exist_ok=True)
        json.dump(cross, open(out, "w"), indent=2)
        print(f"[saved] {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
