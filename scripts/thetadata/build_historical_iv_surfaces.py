from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


DATES = [
    "2021-05-05",
    "2021-05-06",
    "2021-05-07",
    "2021-05-10",
    "2021-05-11",
    "2021-05-12",
    "2021-05-13",
    "2021-05-14",
    "2021-05-17",
    "2021-05-18",
    "2021-05-19",
    "2021-05-20",
    "2021-05-21",
    "2021-05-24",
    "2021-05-25",
    "2021-05-26",
    "2021-05-27",
    "2021-05-28",
    "2021-06-01",
    "2021-06-02",
]

SYMBOL = "SPXW"
QUOTE_TIME = "15:55:00"
TARGET_DTES = "7,14,21,30,45,60,90"

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts/thetadata/build_iv_surface_dataset.py"
OUTDIR = Path("/users/4/trest017/urop_pdv/data/processed/thetadata")


def inspect_outputs(date):
    tag = f"spxw_{date}_155500"

    surface_path = OUTDIR / f"{tag}_iv_surface_dataset.csv"
    summary_path = OUTDIR / f"{tag}_iv_surface_summary.csv"
    failure_path = OUTDIR / f"{tag}_iv_surface_failures.csv"

    row = {
        "trade_date": date,
        "surface_exists": surface_path.exists(),
        "tenors": 0,
        "clean_quotes": 0,
        "failures": 0,
    }

    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            row["tenors"] = len(summary)
            if "clean_iv_count" in summary:
                row["clean_quotes"] = int(
                    summary["clean_iv_count"].sum()
                )
        except Exception:
            pass

    if failure_path.exists() and failure_path.stat().st_size > 1:
        try:
            failures = pd.read_csv(failure_path)
            row["failures"] = len(failures)
        except pd.errors.EmptyDataError:
            pass

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild dates even when output already exists.",
    )
    args = parser.parse_args()

    results = []

    for i, date in enumerate(DATES, start=1):
        tag = f"spxw_{date}_155500"
        surface_path = OUTDIR / f"{tag}_iv_surface_dataset.csv"

        print("\n" + "=" * 78)
        print(f"DAY {i:02d}/{len(DATES)}: {date}")
        print("=" * 78, flush=True)

        returncode = 0

        if surface_path.exists() and not args.force:
            print("Existing surface found; skipping pull.")
        else:
            cmd = [
                sys.executable,
                str(BUILDER),
                "--symbol", SYMBOL,
                "--trade-date", date,
                "--quote-time", QUOTE_TIME,
                "--target-dtes", TARGET_DTES,
            ]

            completed = subprocess.run(cmd)
            returncode = completed.returncode

        row = inspect_outputs(date)
        row["returncode"] = returncode

        if (
            returncode == 0
            and row["surface_exists"]
            and row["tenors"] == 7
            and row["failures"] == 0
        ):
            row["status"] = "OK"
        else:
            row["status"] = "CHECK"

        results.append(row)

    manifest = pd.DataFrame(results)

    manifest_path = (
        OUTDIR
        / "spxw_2021-05-05_2021-06-02_155500_historical_manifest.csv"
    )
    manifest.to_csv(manifest_path, index=False)

    print("\n" + "=" * 78)
    print("HISTORICAL SURFACE MANIFEST")
    print("=" * 78)
    print(manifest.to_string(index=False))
    print("\nWrote:", manifest_path)

    bad = manifest[manifest["status"] != "OK"]

    if len(bad):
        print("\nDates requiring inspection:")
        print(bad.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
