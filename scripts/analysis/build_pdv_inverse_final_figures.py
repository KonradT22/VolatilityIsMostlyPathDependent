from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

MODEL = (
    BASE
    / "models/mlp_interpolation/"
    "test_evaluation"
)

REPRICE = (
    MODEL / "repricing"
)

FRESH = (
    MODEL / "fresh_seed_repricing"
)

GUARD = (
    BASE
    / "guardrails/mahalanobis_chronological"
)

OUT = (
    BASE / "final_figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PARAMS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]


#
# 1. True vs predicted parameters
#
pred = pd.read_csv(
    MODEL / "test_predictions.csv"
)

for name in PARAMS:
    truth = pred[
        f"true_{name}"
    ].to_numpy(float)

    estimate = pred[
        f"clipped_{name}"
    ].to_numpy(float)

    lo = min(
        truth.min(),
        estimate.min(),
    )

    hi = max(
        truth.max(),
        estimate.max(),
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 5.2)
    )

    ax.scatter(
        truth,
        estimate,
        s=12,
        alpha=0.55,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linewidth=1.5,
    )

    ax.set_xlabel(
        f"True {name}"
    )

    ax.set_ylabel(
        f"Predicted {name}"
    )

    ax.set_title(
        f"Inverse PDV Test Recovery: {name}"
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"parameter_recovery_{name}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT
        / f"parameter_recovery_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


#
# 2. Repricing error by maturity
#
by_dte = pd.read_csv(
    REPRICE / "repricing_by_maturity.csv"
)

fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

ax.plot(
    by_dte["target_dte"],
    by_dte["rmse_bp"],
    marker="o",
)

ax.set_xlabel(
    "Option maturity (days)"
)

ax.set_ylabel(
    "RMSE (bp of forward)"
)

ax.set_title(
    "Inverse-ANN Surface Reconstruction Error"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "repricing_rmse_by_maturity.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "repricing_rmse_by_maturity.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


#
# 3. Fresh-seed robustness
#
fresh_seed = pd.read_csv(
    FRESH / "fresh_seed_by_seed.csv"
)

fig, ax = plt.subplots(
    figsize=(6.5, 4.5)
)

labels = [
    str(x)
    for x in fresh_seed["seed"]
]

x = np.arange(
    len(labels)
)

ax.bar(
    x,
    fresh_seed["mean_rmse_bp"],
)

ax.set_xticks(
    x,
    labels,
)

ax.set_xlabel(
    "Independent Monte Carlo seed"
)

ax.set_ylabel(
    "Mean RMSE (bp of forward)"
)

ax.set_title(
    "Fresh-Seed Repricing Robustness"
)

fig.tight_layout()

fig.savefig(
    OUT
    / "fresh_seed_repricing_robustness.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fresh_seed_repricing_robustness.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


#
# 4. 81-D Mahalanobis separation
#
scores = pd.read_csv(
    GUARD
    / "all_81_features_row_scores.csv"
)

id_dist = scores.loc[
    scores["split"]
    == "id_calibration",
    "mahalanobis_distance",
].to_numpy(float)

val_dist = scores.loc[
    scores["split"]
    == "chronological_validation",
    "mahalanobis_distance",
].to_numpy(float)

test_dist = scores.loc[
    scores["split"]
    == "chronological_test",
    "mahalanobis_distance",
].to_numpy(float)

threshold = float(
    np.quantile(
        id_dist,
        .99,
    )
)

fig, ax = plt.subplots(
    figsize=(7.0, 4.8)
)

bins = np.linspace(
    0,
    max(
        id_dist.max(),
        val_dist.max(),
        test_dist.max(),
    ),
    60,
)

ax.hist(
    id_dist,
    bins=bins,
    alpha=0.55,
    label="In-domain calibration",
)

ax.hist(
    val_dist,
    bins=bins,
    alpha=0.55,
    label="Chronological validation",
)

ax.hist(
    test_dist,
    bins=bins,
    alpha=0.55,
    label="Chronological test",
)

ax.axvline(
    threshold,
    linestyle="--",
    linewidth=1.5,
    label=f"ID p99 threshold = {threshold:.2f}",
)

ax.set_xlabel(
    "Squared Mahalanobis distance"
)

ax.set_ylabel(
    "Count"
)

ax.set_title(
    "81-D Input-Space OOD Guardrail"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUT
    / "mahalanobis_81d_ood_separation.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "mahalanobis_81d_ood_separation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("=" * 72)
print("FINAL PDV INVERSE FIGURES")
print("=" * 72)

for p in sorted(
    OUT.glob("*.pdf")
):
    print(p.name)

print()
print("Wrote:", OUT)
