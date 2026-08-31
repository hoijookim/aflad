"""Path Y: EAD-M standalone energy (LeapMind checkpoints, no PUAD overhead)."""
from __future__ import annotations
import sys
import json
import time
import threading
from pathlib import Path
import numpy as np
import torch
import pynvml
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/workspace/ai-vision-research/external/PUAD")
from puad.networks import PDN_M, AutoEncoder
from puad.efficientad.inference import EfficientADInference
from sklearn.metrics import roc_auc_score

REPO = Path("/workspace/ai-vision-research")
LOCO_DIR = REPO / "datasets" / "MVTecLOCO"
LEAPMIND_DIR = REPO / "external" / "PUAD" / "models" / "PUAD" / "mvtec_loco_ad_models" / "mvtec_loco_ad_models"
OUT_DIR = REPO / "reports" / "path_y" / "ead_m_standalone"

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
DEVICE = "cuda"
IMG_SIZE = 256
POLL = 0.05

default_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EnergyTracker:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def start(self):
        self.samples = []; self._running = True; self._t0 = time.time()
        def loop():
            while self._running:
                try:
                    self.samples.append((time.time()-self._t0, pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000))
                except Exception: pass
                time.sleep(POLL)
        self._t = threading.Thread(target=loop, daemon=True); self._t.start()
    def stop(self):
        self._running = False; self._t.join(timeout=1.0)
        if len(self.samples) < 2: return 0.0
        t = np.array([s[0] for s in self.samples])
        p = np.array([s[1] for s in self.samples])
        return float(np.trapezoid(p, t)) if hasattr(np, "trapezoid") else float(np.trapz(p, t))


def list_test(cat):
    base = LOCO_DIR / cat / "test"
    paths = []
    for sub in sorted(base.iterdir()):
        lbl = 0 if sub.name == "good" else 1
        for p in sorted(sub.iterdir()):
            paths.append((p, lbl))
    return paths


def load_ead_m(cat):
    out_channels = 384
    teacher = PDN_M(out_channels=out_channels).to(DEVICE)
    student = PDN_M(out_channels=out_channels * 2).to(DEVICE)
    ae = AutoEncoder(out_channels=out_channels, img_size=IMG_SIZE, padding=False).to(DEVICE)
    size_dir = LEAPMIND_DIR / "m_size"
    cat_dir = size_dir / "mvtec_loco_anomaly_detection" / cat
    teacher.load_state_dict(torch.load(size_dir / "teacher" / "teacher.pt", map_location=DEVICE, weights_only=False))
    student.load_state_dict(torch.load(cat_dir / "student.pt", map_location=DEVICE, weights_only=False))
    ae.load_state_dict(torch.load(cat_dir / "autoencoder.pt", map_location=DEVICE, weights_only=False))
    quantile = torch.load(cat_dir / "quantile.pt", map_location=DEVICE, weights_only=False)
    mu = torch.load(cat_dir / "mu.pt", map_location=DEVICE, weights_only=False)
    sigma = torch.load(cat_dir / "sigma.pt", map_location=DEVICE, weights_only=False)
    return EfficientADInference(teacher=teacher, student=student, autoencoder=ae,
                                  mu=mu, sigma=sigma, quantile=quantile,
                                  img_size=IMG_SIZE, device=DEVICE)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Path Y: EAD-M standalone energy (LeapMind checkpoints)\n")

    per_cat = {}
    for cat in CATS:
        print(f"\n=== {cat} ===")
        ead = load_ead_m(cat)
        test_paths = list_test(cat)
        print(f"  test n={len(test_paths)}")

        # Pre-load
        tensors = []
        labels = []
        for p, lbl in test_paths:
            tensors.append(default_transform(Image.open(p).convert("RGB")))
            labels.append(lbl)

        # Warmup
        for _ in range(3): ead.run(tensors[0])
        torch.cuda.synchronize()

        # Measure
        tracker = EnergyTracker(); tracker.start()
        t0 = time.time(); lat = []; scores = []
        for img in tensors:
            ti = time.time()
            scores.append(float(ead.run(img)))
            torch.cuda.synchronize()
            lat.append((time.time()-ti)*1000)
        elapsed = time.time()-t0
        e = tracker.stop()
        n = len(tensors)
        auroc = roc_auc_score(labels, scores)
        per_cat[cat] = {
            "n_test": n,
            "energy_per_img_mJ": e * 1000 / n,
            "median_latency_ms": float(np.median(lat)),
            "fps": n / elapsed,
            "ead_m_auroc": float(auroc),
        }
        print(f"  Energy: {per_cat[cat]['energy_per_img_mJ']:.1f} mJ, Lat: {per_cat[cat]['median_latency_ms']:.2f} ms, AUROC: {auroc:.4f}")
        del ead
        torch.cuda.empty_cache()

    energies = [v["energy_per_img_mJ"] for v in per_cat.values()]
    aurocs = [v["ead_m_auroc"] for v in per_cat.values()]
    summary = {
        "method": "EAD-M standalone (LeapMind m_size, no PUAD Mahalanobis)",
        "per_cat": per_cat,
        "aggregate": {
            "energy_per_img_mJ_mean": float(np.mean(energies)),
            "energy_per_img_mJ_std": float(np.std(energies)),
            "ead_m_auroc_mean": float(np.mean(aurocs)),
            "paper_ead_m_mean": 0.9099,
            "auroc_gap_pp": float((np.mean(aurocs) - 0.9099) * 100),
        },
    }
    print("\n========== Aggregate ==========")
    print(f"EAD-M energy: {summary['aggregate']['energy_per_img_mJ_mean']:.1f} ± {summary['aggregate']['energy_per_img_mJ_std']:.1f} mJ")
    print(f"EAD-M AUROC: {summary['aggregate']['ead_m_auroc_mean']:.4f} (paper 0.9099, gap {summary['aggregate']['auroc_gap_pp']:+.2f}pp)")
    out = OUT_DIR / "ead_m_standalone.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] saved {out}")


if __name__ == "__main__":
    main()
