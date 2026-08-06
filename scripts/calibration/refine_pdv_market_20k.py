import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import scripts.calibration.calibrate_pdv_market_multi_maturity as market


# Warm-start from the robust bounded 4k calibration.
market.START_BETAS = np.array(
    [
        0.05996444574148981,
        -0.14386034079634102,
        0.6180842038647475,
    ],
    dtype=float,
)

# Higher-fidelity objective.
market.N_PATHS = 20000

# Use a fresh calibration seed, separate from the five seeds
# already used for validation.
market.SEED_ROOT = 2026080610

# Retain the original bounded parameter domain because the
# expanded-beta0 solution did not generalize better.
market.PARAMETER_BOUNDS["beta0"] = (
    0.02,
    0.06,
)

market.OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_20k_refinement"
)


if __name__ == "__main__":
    market.main()
