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
LANGUAGE_COLUMN = "lang_detected"
SPANISH_LABEL = "es"

# Integers on a 1-5 scale, so any movement counts. PPL is handled apart: it is
# unbounded and lower is better, so it needs a relative tolerance instead.
LIKERT_COLUMNS = ("naturalness", "meaningfulness")
DEFAULT_PPL_TOLERANCE = 0.10

# Regressions first: what a rung broke is more interesting than what it fixed.
VERDICT_ORDER = ("regressed", "mixed", "improved", "unchanged", "unknown")
VERDICT_SYMBOLS = {
    "improved": "+",
    "regressed": "-",
    "mixed": "~",
    "unchanged": "·",
    "unknown": "?",
}
# A prompt the two runs do not share is not the same as one with no comparable
# signal, so it gets its own (empty) cell rather than reusing `unknown`.
ABSENT_SYMBOL = " "
MATRIX_LEGEND = (
    "+ improved   - regressed   ~ mixed   · unchanged   "
    "? no comparable signal   (blank) not in both runs"
)


def share_spanish(labels: pd.Series) -> float:
    known = labels.dropna().astype(str).str.strip()
    known = known[known != ""]
    if known.empty:
        return float("nan")
    return 100.0 * float((known == SPANISH_LABEL).mean())


def clean_label(value) -> str | None:
    """A language label, or None when the run has no usable one."""
    if pd.isna(value):
        return None
    return str(value).strip() or None


def language_change(before, after) -> str:
    before, after = clean_label(before), clean_label(after)
    if before is None or after is None:
        return ""
    return f"{before}→{after}"


def coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in METRIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = float("nan")
    if LANGUAGE_COLUMN not in out.columns:
        out[LANGUAGE_COLUMN] = None
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    numeric = coerce_metrics(df)
    out = numeric.groupby(["run_id", "block"], dropna=False).agg(
        n=("prompt_id", "size"),
        mean_naturalness=("naturalness", "mean"),
        mean_meaningfulness=("meaningfulness", "mean"),
        mean_ppl=("ppl", "mean"),
        median_ppl=("ppl", "median"),
        pct_spanish=(LANGUAGE_COLUMN, share_spanish),
    )
    return out.reset_index()


def classify(row: pd.Series, ppl_tolerance: float) -> str:
    """Whether this prompt got better, worse, or neither, across all signals.

    Counts how many signals moved each way rather than combining them into a
    score: the four are on incomparable scales, and "improved on two, broke on
    one" is a fact worth surfacing as `mixed` instead of averaging away.
    """
    good = bad = known = 0

    for metric in LIKERT_COLUMNS:
        current, baseline = row[metric], row[f"{metric}_baseline"]
        if pd.isna(current) or pd.isna(baseline):
            continue
        known += 1
        if current > baseline:
            good += 1
        elif current < baseline:
            bad += 1

    current, baseline = row["ppl"], row["ppl_baseline"]
    if not pd.isna(current) and not pd.isna(baseline) and baseline > 0:
        known += 1
        relative = (current - baseline) / baseline
        if relative < -ppl_tolerance:
            good += 1
        elif relative > ppl_tolerance:
            bad += 1

    current = clean_label(row[LANGUAGE_COLUMN])
    baseline = clean_label(row["lang_baseline"])
    if current is not None and baseline is not None:
        known += 1
        if current == SPANISH_LABEL and baseline != SPANISH_LABEL:
            good += 1
        elif baseline == SPANISH_LABEL and current != SPANISH_LABEL:
            bad += 1

    # Not the same as `unchanged`: nothing was comparable at all. Happens before
    # the human pass, when only the automatic columns are filled in.
    if known == 0:
        return "unknown"
    if good and bad:
        return "mixed"
    if good:
        return "improved"
    if bad:
        return "regressed"
    return "unchanged"


PER_PROMPT_COLUMNS = [
    "prompt_id",
    "block",
    "run_id",
    "baseline_run_id",
    "verdict",
    "naturalness",
    "naturalness_baseline",
    "naturalness_delta",
    "meaningfulness",
    "meaningfulness_baseline",
    "meaningfulness_delta",
    "ppl",
    "ppl_baseline",
    "ppl_delta",
    "ppl_rel_delta",
    LANGUAGE_COLUMN,
    "lang_baseline",
    "lang_change",
]


def compare_runs(
    df: pd.DataFrame, baseline_id: str, ppl_tolerance: float
) -> pd.DataFrame:
    """One row per (prompt, run), each run compared against the baseline run."""
    numeric = coerce_metrics(df)
    baseline = numeric[numeric["run_id"] == baseline_id]
    baseline = baseline[["prompt_id", *METRIC_COLUMNS, LANGUAGE_COLUMN]].rename(
        columns={
            **{metric: f"{metric}_baseline" for metric in METRIC_COLUMNS},
            LANGUAGE_COLUMN: "lang_baseline",
        }
    )

    frames: list[pd.DataFrame] = []
    for order, run_id in enumerate(numeric["run_id"].drop_duplicates()):
        if run_id == baseline_id:
            continue
        current = numeric[numeric["run_id"] == run_id]
        merged = current.merge(baseline, on="prompt_id", how="inner")

        dropped = set(current["prompt_id"]) ^ set(baseline["prompt_id"])
        if dropped:
            logger.warning(
                "%s vs %s: %d prompt(s) only present in one of the two runs, "
                "excluded from the diff: %s",
                run_id,
                baseline_id,
                len(dropped),
                ", ".join(sorted(dropped)),
            )
        if merged.empty:
            logger.warning("%s vs %s: no prompts in common", run_id, baseline_id)
            continue

        for metric in METRIC_COLUMNS:
            merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_baseline"]
        merged["ppl_rel_delta"] = merged["ppl_delta"] / merged["ppl_baseline"]
        merged["lang_change"] = [
            language_change(before, after)
            for before, after in zip(merged["lang_baseline"], merged[LANGUAGE_COLUMN])
        ]
        merged["verdict"] = merged.apply(classify, axis=1, ppl_tolerance=ppl_tolerance)
        merged["baseline_run_id"] = baseline_id
        merged["_run_order"] = order
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=PER_PROMPT_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["_rank"] = out["verdict"].map({v: i for i, v in enumerate(VERDICT_ORDER)})
    out = out.sort_values(["_rank", "_run_order", "prompt_id"])
    return out[PER_PROMPT_COLUMNS].reset_index(drop=True)


def format_matrix(
    per_prompt: pd.DataFrame, prompt_order: list[str], width: int = 100
) -> str:
    """The verdicts as a grid, wrapped into bands so it fits a terminal."""
    comparisons = per_prompt[["run_id", "baseline_run_id"]].drop_duplicates()
    labels = [f"{row.run_id} vs {row.baseline_run_id}" for row in comparisons.itertuples()]
    label_width = max(len(label) for label in labels)

    verdicts = {
        (row.run_id, row.prompt_id): row.verdict for row in per_prompt.itertuples()
    }
    shown = [p for p in prompt_order if p in set(per_prompt["prompt_id"])]
    cell_width = max(len(p) for p in shown) + 1
    per_band = max(1, (width - label_width - 1) // cell_width)

    lines: list[str] = []
    for start in range(0, len(shown), per_band):
        band = shown[start : start + per_band]
        lines.append(
            " " * (label_width + 1) + "".join(p.ljust(cell_width) for p in band)
        )
        for label, row in zip(labels, comparisons.itertuples()):
            cells = "".join(
                VERDICT_SYMBOLS.get(verdicts.get((row.run_id, p)), ABSENT_SYMBOL)
                .center(len(p))
                .ljust(cell_width)
                for p in band
            )
            lines.append(f"{label.ljust(label_width)} {cells}")
        lines.append("")
    return "\n".join(lines).rstrip()


def summarize(per_prompt: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for (run_id, baseline_id), group in per_prompt.groupby(
        ["run_id", "baseline_run_id"], sort=False
    ):
        counts = group["verdict"].value_counts()
        parts = [
            f"{int(counts.get('improved', 0))} improved",
            f"{int(counts.get('regressed', 0))} regressed",
            f"{int(counts.get('unchanged', 0))} unchanged",
        ]
        for extra in ("mixed", "unknown"):
            if counts.get(extra, 0):
                parts.append(f"{int(counts[extra])} {extra}")
        to_spanish = sum(
            1
            for change in group["lang_change"]
            if change and change.endswith(f"→{SPANISH_LABEL}")
            and not change.startswith(f"{SPANISH_LABEL}→")
        )
        lines.append(
            f"{run_id} vs {baseline_id}: {', '.join(parts)}"
            f" | {to_spanish} switched to Spanish"
        )
    return lines


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Aggregate annotations + auto-metrics across several runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="Paths to runs/<id>/")
    parser.add_argument("--out", type=Path, default=Path(f"./reports/comparison_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"), help="Output CSV (default: ./reports/comparison_<timestamp>.csv).")
    parser.add_argument("--baseline", type=Path, default=None, help="Run every other run is compared against. Defaults to the first one given.")
    parser.add_argument("--ppl-tolerance", type=float, default=DEFAULT_PPL_TOLERANCE, help=f"Relative PPL change below which a prompt counts as unchanged. Default: {DEFAULT_PPL_TOLERANCE}.")
    args = parser.parse_args()

    dfs = [load_run(p) for p in args.runs]
    df = pd.concat(dfs, ignore_index=True)
    summary = aggregate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    logger.info(f"Written to {args.out}")

    if len(args.runs) < 2:
        logger.info("Only one run given; skipping the per-prompt diff.")
        return 0

    baseline_id = (args.baseline or args.runs[0]).name
    known_runs = list(df["run_id"].drop_duplicates())
    if baseline_id not in known_runs:
        parser.error(f"baseline {baseline_id!r} is not among the runs given: {known_runs}")

    per_prompt = compare_runs(df, baseline_id, args.ppl_tolerance)
    if per_prompt.empty:
        logger.warning("Nothing to compare per prompt.")
        return 0

    per_prompt_path = args.out.with_name(f"{args.out.stem}_per_prompt{args.out.suffix}")
    per_prompt.to_csv(per_prompt_path, index=False)
    logger.info(f"Written to {per_prompt_path}")

    prompt_order = list(df["prompt_id"].drop_duplicates())
    print(format_matrix(per_prompt, prompt_order))
    print(MATRIX_LEGEND)
    print()
    for line in summarize(per_prompt):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
