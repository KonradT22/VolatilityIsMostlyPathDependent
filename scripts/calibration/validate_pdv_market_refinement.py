import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import scripts.calibration.validate_pdv_market_calibrations as validation


validation.OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_refinement_validation"
)

validation.VALIDATION_N = 20000

# Completely new seeds:
# - not the 4k calibration seed
# - not the 20k refinement seed
# - not the previous validation seeds
validation.SEEDS = [
    2026080620,
    2026080621,
    2026080622,
    2026080623,
    2026080624,
]

validation.CANDIDATES = {
    "baseline": np.array(
        [
            0.04,
            -0.13,
            0.65,
        ],
        dtype=float,
    ),

    "bounded_4k_calibration": np.array(
        [
            0.05996444574148981,
            -0.14386034079634102,
            0.6180842038647475,
        ],
        dtype=float,
    ),

    "refined_20k_calibration": np.array(
        [
            0.03649376451669183,
            -0.1285527875517993,
            0.7375579398479576,
        ],
        dtype=float,
    ),
}


if __name__ == "__main__":
    validation.main()
