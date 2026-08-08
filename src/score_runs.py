from __future__ import annotations

import argparse
import datetime as dt
import logging
import logging.config
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_run(run_dir: Path) -> pd.DataFrame:
    ann_path = run_dir / "annotations.csv"
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} does not exist.")
    df = pd.read_csv(ann_path)
    auto_path = run_dir / "auto_metrics.csv"
    if auto_path.exists():
        auto = pd.read_csv(auto_path)
        auto = auto.drop(columns=["block"], errors="ignore")
        df = df.merge(auto, on="prompt_id", how="left")
    df["run_id"] = run_dir.name
    return df


METRIC_COLUMNS = ("naturalness", "meaningfulness", "ppl")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for col in METRIC_COLUMNS:
        if col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        else:
            numeric[col] = float("nan")

    out = numeric.groupby(["run_id", "block"], dropna=False).agg(
        n=("prompt_id", "size"),
        mean_naturalness=("naturalness", "mean"),
        mean_meaningfulness=("meaningfulness", "mean"),
        mean_ppl=("ppl", "mean"),
        median_ppl=("ppl", "median"),
    )
    return out.reset_index()


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Aggregate annotations + auto-metrics across several runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="Paths to runs/<id>/")
    parser.add_argument("--out", type=Path, default=Path(f"./reports/comparison_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"), help="Output CSV (default: ./reports/comparison_<timestamp>.csv).")
    args = parser.parse_args()

    dfs = [load_run(p) for p in args.runs]
    df = pd.concat(dfs, ignore_index=True)
    summary = aggregate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    logger.info(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
