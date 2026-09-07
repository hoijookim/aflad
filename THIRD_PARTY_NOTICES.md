# Third-party components

This repository redistributes derived artifacts (per-image anomaly scores) only.
No third-party model weights or datasets are included. The pipeline builds on:

- EfficientAD (official implementation) — reconstruction branch
- PatchCore — patch-memory branch formulation
- DINOv3-L/16 (Meta) — frozen backbone; distributor gated license applies
- PSAD (official implementation) — composition branch
- CSAD (Tokichan/CSAD, commit d46cf85) — component pseudo-labels
- MVTec LOCO AD (MVTec Software GmbH) — dataset, redistributed by the original
  rights holder only; not included here

Each component remains under its own license. Users must obtain the dataset and
model weights from their original distributors.
