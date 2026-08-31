"""PUAD-M multi-seed inference (3 seeds × 5 cats).

Uses checkpoints from:
  reports/phase0/efficient_ad_official_medium/{cat}_seed42_v1/
  reports/phase0/efficient_ad_official_medium_seeds/{cat}_seed43/
  reports/phase0/efficient_ad_official_medium_seeds/{cat}_seed44/

Pipeline identical to PY_real_puad_M_FULL.py per seed. Aggregates 3-seed mean ± std.
"""
from __future__ import annotations
import json
import time
import threading
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pynvml
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

REPO = Path("/workspace/ai-vision-research")
OUT_DIR = REPO / "reports" / "path_y" / "energy"
LOCO_DIR = REPO / "datasets" / "MVTecLOCO"

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
POLL = 0.05
IMG_SIZE = 256
OUT_CHANNELS = 384
DEVICE = "cuda"

# Per (seed, cat) ckpt path
def ckpt_dir(seed, cat):
    if seed == 42:
        return REPO / "reports/phase0/efficient_ad_official_medium" / f"{cat}_seed42_v1" / "trainings/mvtec_loco" / cat
    return REPO / "reports/phase0/efficient_ad_official_medium_seeds" / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat


default_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_models(seed, cat):
    cd = ckpt_dir(seed, cat)
    teacher = torch.load(cd / "teacher_final.pth", map_location=DEVICE, weights_only=False)
    student = torch.load(cd / "student_final.pth", map_location=DEVICE, weights_only=False)
    ae = torch.load(cd / "autoencoder_final.pth", map_location=DEVICE, weights_only=False)
    teacher.eval(); student.eval(); ae.eval()
    return teacher, student, ae


def load_image(path):
    return default_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)


def list_split(cat, split):
    base = LOCO_DIR / cat / split
    paths = []
    if split in ("train", "validation"):
        for p in sorted((base / "good").iterdir()):
            paths.append((p, 0))
    else:
        for sub in sorted(base.iterdir()):
            lbl = 0 if sub.name == "good" else 1
            for p in sorted(sub.iterdir()):
                paths.append((p, lbl))
    return paths


@torch.no_grad()
def teacher_normalization(teacher, train_paths):
    mean_outputs = []
    for p, _ in train_paths:
        t = teacher(load_image(p))
        mean_outputs.append(t.mean(dim=[0, 2, 3]))
    cm = torch.stack(mean_outputs).mean(dim=0)[None, :, None, None]
    mean_dists = []
    for p, _ in train_paths:
        t = teacher(load_image(p))
        mean_dists.append(((t - cm) ** 2).mean(dim=[0, 2, 3]))
    cv = torch.stack(mean_dists).mean(dim=0)[None, :, None, None]
    return cm, torch.sqrt(cv)


@torch.no_grad()
def predict_raw_maps(image, teacher, student, ae, t_mean, t_std):
    t_out = (teacher(image) - t_mean) / t_std
    s_out = student(image)
    a_out = ae(image)
    map_st = ((t_out - s_out[:, :OUT_CHANNELS]) ** 2).mean(dim=1, keepdim=True)
    map_ae = ((a_out - s_out[:, OUT_CHANNELS:]) ** 2).mean(dim=1, keepdim=True)
    return map_st, map_ae, s_out[:, :OUT_CHANNELS]


@torch.no_grad()
def predict_full(image, teacher, student, ae, t_mean, t_std,
                 q_st_s, q_st_e, q_ae_s, q_ae_e):
    map_st, map_ae, s_first = predict_raw_maps(image, teacher, student, ae, t_mean, t_std)
    map_st = F.interpolate(map_st, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    map_ae = F.interpolate(map_ae, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    map_st = 0.1 * (map_st - q_st_s) / (q_st_e - q_st_s)
    map_ae = 0.1 * (map_ae - q_ae_s) / (q_ae_e - q_ae_s)
    map_comb = 0.5 * map_st + 0.5 * map_ae
    return map_comb.max().item(), s_first.mean(dim=(0, 2, 3)).cpu().numpy()


@torch.no_grad()
def map_normalization(val_paths, teacher, student, ae, t_mean, t_std):
    maps_st, maps_ae = [], []
    for p, _ in val_paths:
        m_st, m_ae, _ = predict_raw_maps(load_image(p), teacher, student, ae, t_mean, t_std)
        maps_st.append(F.interpolate(m_st, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False))
        maps_ae.append(F.interpolate(m_ae, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False))
    ms = torch.cat(maps_st); ma = torch.cat(maps_ae)
    return torch.quantile(ms, 0.9), torch.quantile(ms, 0.995), torch.quantile(ma, 0.9), torch.quantile(ma, 0.995)


def fit_puad_train(train_paths, teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e):
    feats, scores = [], []
    for p, _ in train_paths:
        s, f = predict_full(load_image(p), teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e)
        feats.append(f); scores.append(s)
    feats = np.array(feats)
    return feats.mean(0), np.linalg.pinv(LedoitWolf().fit(feats).covariance_), float(np.mean(scores)), float(np.std(scores))


def fit_puad_val(val_paths, teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e, mean_v, cov_inv):
    eads, mahas = [], []
    for p, _ in val_paths:
        s, f = predict_full(load_image(p), teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e)
        eads.append(s)
        c = f - mean_v
        mahas.append(float(np.sqrt(max(0.0, c @ cov_inv @ c))))
    return float(np.mean(eads)), float(np.std(eads)), float(np.mean(mahas)), float(np.std(mahas))


class EnergyTracker:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def start(self):
        self.samples = []; self._t0 = time.time(); self._running = True
        def _loop():
            while self._running:
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.samples.append((time.time() - self._t0, p))
                except Exception: pass
                time.sleep(POLL)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
    def stop(self):
        self._running = False; self._thread.join(timeout=1.0)
        if len(self.samples) < 2: return 0.0
        t = np.array([s[0] for s in self.samples]); p = np.array([s[1] for s in self.samples])
        return float(np.trapezoid(p, t)) if hasattr(np, "trapezoid") else float(np.trapz(p, t))


def measure(teacher, student, ae, test_paths, t_mean, t_std,
            q_st_s, q_st_e, q_ae_s, q_ae_e,
            mean_v, cov_inv, ead_mu, ead_sig, maha_mu, maha_sig, mode="puad"):
    tensors = []; labels = []
    for p, lbl in test_paths:
        tensors.append(load_image(p)); labels.append(lbl)
    for _ in range(3):
        predict_full(tensors[0], teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e)
    torch.cuda.synchronize()
    tracker = EnergyTracker(); tracker.start()
    t0 = time.time(); latencies = []; scores = []
    for img in tensors:
        ti = time.time()
        s, f = predict_full(img, teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e)
        if mode == "puad":
            c = f - mean_v
            maha = float(np.sqrt(max(0.0, c @ cov_inv @ c)))
            final = (s - ead_mu) / max(ead_sig, 1e-6) + (maha - maha_mu) / max(maha_sig, 1e-6)
        else:
            final = s
        torch.cuda.synchronize()
        latencies.append((time.time() - ti) * 1000); scores.append(final)
    elapsed = time.time() - t0
    energy_J = tracker.stop()
    n = len(tensors)
    try:
        auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    return {"n_images": n, "energy_per_img_mJ": energy_J * 1000 / n,
            "median_latency_ms": float(np.median(latencies)),
            "fps": n / elapsed, "auroc": float(auroc),
            "scores": [float(s) for s in scores], "labels": [int(l) for l in labels]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("PUAD-M multi-seed inference (3 seeds × 5 cats)\n")

    per_seed = {}
    for seed in [42, 43, 44]:
        per_seed[seed] = {}
        for cat in CATS:
            cd = ckpt_dir(seed, cat)
            if not (cd / "teacher_final.pth").exists():
                print(f"[skip] seed={seed} {cat} ckpt missing at {cd}")
                continue
            print(f"\n=== seed={seed} / {cat} ===")
            teacher, student, ae = load_models(seed, cat)
            train_paths = list_split(cat, "train")
            val_paths = list_split(cat, "validation")
            test_paths = list_split(cat, "test")
            t_mean, t_std = teacher_normalization(teacher, train_paths)
            q_st_s, q_st_e, q_ae_s, q_ae_e = map_normalization(val_paths, teacher, student, ae, t_mean, t_std)
            mean_v, cov_inv, _, _ = fit_puad_train(train_paths, teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e)
            ead_mu, ead_sig, maha_mu, maha_sig = fit_puad_val(val_paths, teacher, student, ae, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e, mean_v, cov_inv)
            ead = measure(teacher, student, ae, test_paths, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e, mean_v, cov_inv, ead_mu, ead_sig, maha_mu, maha_sig, mode="ead")
            puad = measure(teacher, student, ae, test_paths, t_mean, t_std, q_st_s, q_st_e, q_ae_s, q_ae_e, mean_v, cov_inv, ead_mu, ead_sig, maha_mu, maha_sig, mode="puad")
            per_seed[seed][cat] = {"ead_m_only": {k: v for k, v in ead.items() if k not in ("scores", "labels")},
                                    "puad_m": {k: v for k, v in puad.items() if k not in ("scores", "labels")}}
            sd = REPO / "reports" / "path_y" / "puad_m_multiseed_scores"
            sd.mkdir(parents=True, exist_ok=True)
            np.savez(sd / f"{cat}_seed{seed}.npz",
                     ead=np.array(ead["scores"]), puad=np.array(puad["scores"]),
                     labels=np.array(puad["labels"]))
            print(f"  EAD-M:  {ead['energy_per_img_mJ']:.1f} mJ, AUROC {ead['auroc']:.4f}")
            print(f"  PUAD-M: {puad['energy_per_img_mJ']:.1f} mJ, AUROC {puad['auroc']:.4f}")

    # Aggregate per cat across seeds
    print("\n========== Per-cat 3-seed aggregate ==========")
    per_cat_agg = {}
    for cat in CATS:
        ead_aurocs = [per_seed[s][cat]["ead_m_only"]["auroc"] for s in [42,43,44] if cat in per_seed.get(s, {})]
        puad_aurocs = [per_seed[s][cat]["puad_m"]["auroc"] for s in [42,43,44] if cat in per_seed.get(s, {})]
        ead_energies = [per_seed[s][cat]["ead_m_only"]["energy_per_img_mJ"] for s in [42,43,44] if cat in per_seed.get(s, {})]
        puad_energies = [per_seed[s][cat]["puad_m"]["energy_per_img_mJ"] for s in [42,43,44] if cat in per_seed.get(s, {})]
        per_cat_agg[cat] = {
            "ead_m_auroc_mean": float(np.mean(ead_aurocs)), "ead_m_auroc_std": float(np.std(ead_aurocs)),
            "puad_m_auroc_mean": float(np.mean(puad_aurocs)), "puad_m_auroc_std": float(np.std(puad_aurocs)),
            "ead_m_energy_mean_mJ": float(np.mean(ead_energies)),
            "puad_m_energy_mean_mJ": float(np.mean(puad_energies)),
            "ead_m_auroc_per_seed": ead_aurocs, "puad_m_auroc_per_seed": puad_aurocs,
        }
        print(f"  {cat:<22} EAD-M {per_cat_agg[cat]['ead_m_auroc_mean']:.4f} ± {per_cat_agg[cat]['ead_m_auroc_std']:.4f} | PUAD-M {per_cat_agg[cat]['puad_m_auroc_mean']:.4f} ± {per_cat_agg[cat]['puad_m_auroc_std']:.4f}")

    # 5-cat aggregate (per-seed mean then average)
    seed_means_ead = []
    seed_means_puad = []
    for s in [42, 43, 44]:
        if s in per_seed and len(per_seed[s]) == 5:
            seed_means_ead.append(np.mean([per_seed[s][c]["ead_m_only"]["auroc"] for c in CATS]))
            seed_means_puad.append(np.mean([per_seed[s][c]["puad_m"]["auroc"] for c in CATS]))

    summary = {
        "method": "PUAD-M multi-seed FULL reproduction",
        "seeds": [42, 43, 44],
        "per_seed": per_seed,
        "per_cat_3seed": per_cat_agg,
        "5cat_aggregate": {
            "ead_m_5cat_mean_per_seed": [float(x) for x in seed_means_ead],
            "ead_m_overall": float(np.mean(seed_means_ead)),
            "ead_m_overall_std": float(np.std(seed_means_ead)),
            "puad_m_5cat_mean_per_seed": [float(x) for x in seed_means_puad],
            "puad_m_overall": float(np.mean(seed_means_puad)),
            "puad_m_overall_std": float(np.std(seed_means_puad)),
        },
        "paper_targets": {"puad_s_paper": 0.9318, "puad_m_paper": 0.9444},
    }
    print(f"\n========== 5-cat 3-seed overall ==========")
    print(f"  EAD-M:  {summary['5cat_aggregate']['ead_m_overall']:.4f} ± {summary['5cat_aggregate']['ead_m_overall_std']:.4f}  (per-seed {summary['5cat_aggregate']['ead_m_5cat_mean_per_seed']})")
    print(f"  PUAD-M: {summary['5cat_aggregate']['puad_m_overall']:.4f} ± {summary['5cat_aggregate']['puad_m_overall_std']:.4f}  (per-seed {summary['5cat_aggregate']['puad_m_5cat_mean_per_seed']})")
    print(f"  vs paper PUAD-M 0.9444: Δ {summary['5cat_aggregate']['puad_m_overall'] - 0.9444:+.4f}")

    out = OUT_DIR / "real_puad_m_multiseed.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] saved {out}")


if __name__ == "__main__":
    main()
