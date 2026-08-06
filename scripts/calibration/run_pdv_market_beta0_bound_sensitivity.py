import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import scripts.calibration.calibrate_pdv_market_multi_maturity as market


# Warm-start from the previously calibrated market solution.
market.START_BETAS = np.array(
    [
        0.05996444574148981,
        -0.14386034079634102,
        0.6180842038647475,
    ],
    dtype=float,
)

# The original [0.02, 0.06] beta0 range came from our
# pilot surrogate experiments, not a theoretical PDV restriction.
#
# Expand only beta0 so this experiment isolates the effect
# of that boundary.
market.PARAMETER_BOUNDS["beta0"] = (
    0.02,
    0.10,
)

market.OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_beta0_bound_sensitivity"
)


if __name__ == "__main__":
    market.main()
