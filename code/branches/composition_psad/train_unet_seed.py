#!/usr/bin/env python3
"""train_normal_unet.py 시드 래퍼 (run_eads_seed.py 패턴).

공식 스크립트는 --random_seed 로 torch 만 시드하고 python random(회전 증강)은
시드하지 않는다. 재현성을 위해 3종을 모두 심고 main() 을 호출한다.

사용: python3 train_unet_seed.py <seed> --obj_name breakfast_box [기타 인자...]
"""
import os
import random
import sys

import numpy as np
import torch

if len(sys.argv) < 2:
    print("사용: train_unet_seed.py <seed> [train_normal_unet 인자...]", file=sys.stderr)
    sys.exit(2)

seed = int(sys.argv.pop(1))

SHIMS = "/workspace/ai-vision-research/scripts/psad_rebuild/_shims"
REPO = "/workspace/ai-vision-research/external/PSAD_official"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.append(SHIMS)  # apex 스텁(미사용 import 우회)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sys.argv += ["--random_seed", str(seed)]

import train_normal_unet  # noqa: E402

train_normal_unet.main()
