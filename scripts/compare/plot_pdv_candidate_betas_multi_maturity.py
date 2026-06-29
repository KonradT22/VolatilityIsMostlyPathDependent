from pathlib import Path
import argparse

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_LABELS = {
    "baseline": "Baseline [0.04, -0.13, 0.65]",
    "best_30d_beta1_m009_beta2_075": "Best 30D [0.04, -0.09, 0.75]",
    "slope_balanced_beta1_m010_beta2_075": "Slope-balanced [0.04, -0.10, 0.75]",
}


def plot_metric(df, metric_name, market_col, pdv_col, ylabel, title, out_path):
    plt.figure()

    market = df[["dte", market_col]].drop_duplicates().sort_values("dte")
    plt.plot(
        market["dte"],
        market[market_col],
        marker="o",
        linewidth=2,
        label="Market",
    )

    for case_name, group in df.groupby("case_name"):
        group = group.sort_values("dte")
        label = CASE_LABELS.get(case_name, case_name)
        plt.plot(
            group["dte"],
            group[pdv_col],
            marker="o",
            label=label,
        )

    plt.xlabel("DTE")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("Wrote:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline and candidate PDV beta fits against market targets."
    )
    parser.add_argument(
        "--input",
        default="/users/4/trest017/urop_pdv/benchmarks/pdv_candidate_betas/pdv_candidate_betas_multi_maturity_vs_market.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/users/4/trest017/urop_pdv/benchmarks/pdv_candidate_betas",
    )

    args = parser.parse_args()

    infile = Path(args.input)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    plot_metric(
        df,
        metric_name="atm_iv",
        market_col="atm_iv",
        pdv_col="pdv_atm_iv",
        ylabel="ATM implied volatility",
        title="ATM IV: PDV Candidates vs Market",
        out_path=outdir / "pdv_candidate_betas_atm_iv.png",
    )

    plot_metric(
        df,
        metric_name="left_5pct_iv",
        market_col="left_5pct_iv",
        pdv_col="pdv_left_5pct_iv",
        ylabel="Left 5% implied volatility",
        title="Left-Wing IV: PDV Candidates vs Market",
        out_path=outdir / "pdv_candidate_betas_left_5pct_iv.png",
    )

    plot_metric(
        df,
        metric_name="right_5pct_iv",
        market_col="right_5pct_iv",
        pdv_col="pdv_right_5pct_iv",
        ylabel="Right 5% implied volatility",
        title="Right-Wing IV: PDV Candidates vs Market",
        out_path=outdir / "pdv_candidate_betas_right_5pct_iv.png",
    )

    plot_metric(
        df,
        metric_name="wing_gap",
        market_col="wing_iv_gap_left_minus_right",
        pdv_col="pdv_wing_iv_gap_left_minus_right",
        ylabel="Left minus right IV",
        title="Wing Gap: PDV Candidates vs Market",
        out_path=outdir / "pdv_candidate_betas_wing_gap.png",
    )

    plot_metric(
        df,
        metric_name="skew_slope",
        market_col="atm_skew_slope",
        pdv_col="pdv_atm_skew_slope",
        ylabel="ATM skew slope",
        title="ATM Skew Slope: PDV Candidates vs Market",
        out_path=outdir / "pdv_candidate_betas_skew_slope.png",
    )

    plt.figure()
    for case_name, group in df.groupby("case_name"):
        group = group.sort_values("dte")
        label = CASE_LABELS.get(case_name, case_name)
        plt.plot(
            group["dte"],
            group["objective_no_slope"],
            marker="o",
            label=label,
        )

    plt.xlabel("DTE")
    plt.ylabel("Objective without slope")
    plt.title("Per-Maturity Objective: PDV Candidates")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    objective_path = outdir / "pdv_candidate_betas_objective_no_slope.png"
    plt.savefig(objective_path, dpi=150)
    print("Wrote:", objective_path)

    print()
    print("Loaded:", infile)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
