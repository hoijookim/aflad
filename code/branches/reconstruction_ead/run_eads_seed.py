"""Wrapper: run nelson1425 EAD with overridden seed (script hardcodes 42).

Usage: python3.12 run_eads_seed.py <seed> [efficientad.py args...]
"""
import os
import sys
import random
import numpy as np
import torch

if len(sys.argv) < 2:
    print("Usage: run_eads_seed.py <seed> [efficientad args...]", file=sys.stderr)
    sys.exit(2)

seed = int(sys.argv.pop(1))  # pop seed; argv[0] stays as script name

REPO = "/workspace/ai-vision-research/external/efficient_ad_official"
os.chdir(REPO)
sys.path.insert(0, REPO)

# Override seed globally
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Now import and run efficientad (it will re-seed itself with hardcoded 42; we re-override after import)
import efficientad as ead
ead.seed = seed  # override module-level
# Patch main to use our seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ead.main()
