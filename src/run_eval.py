from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import logging.config
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.models.factory import load_model

logger = logging.getLogger(__name__)

ANNOTATION_COLUMNS = [
    "prompt_id",
    "block",
    "naturalness",
    "meaningfulness",
    "notes",
]


@dataclass
class RunConfig:
    model_type: str
    tag: str
    checkpoint: str
    temperature_audio: float
    temperature_text: float
    top_k_audio: int
    top_k_text: int
    cfg_coef: float
    seed: int
    device: str
    lora_weight: str | None = None
    dtype: str | None = None


def group_into_sessions(bank: list[dict]) -> list[list[dict]]:
    sessions: list[list[dict]] = []
    current: list[dict] = []
    current_seq: str | None = None
    for entry in bank:
        seq = entry.get("sequence")
        if seq is None:
            if current:
                sessions.append(current)
                current = []
                current_seq = None
            sessions.append([entry])
        elif seq == current_seq:
            current.append(entry)
        else:
            if current:
                sessions.append(current)
            current = [entry]
            current_seq = seq
    if current:
        sessions.append(current)
    return sessions


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Run a speech model over the prompt bank and save outputs in runs/.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the run's config.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Create the run dir and annotations.csv without loading the model.")
    parser.add_argument("--bank", type=Path, default=Path("./prompts/bank.yaml"), help="Path to the prompt bank.")
    parser.add_argument("--wav", type=Path, default=Path("./prompts/wav"), help="Path to the wav output dir.")
    args = parser.parse_args()

    with args.config.open() as f:
        data = yaml.safe_load(f)
        cfg = RunConfig(**data)

    with args.bank.open() as f:
        bank = yaml.safe_load(f)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(f"./runs/{stamp}_{cfg.tag}_seed{cfg.seed}")
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(args.config.read_text())

    csv_path = run_dir / "annotations.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for entry in bank:
            writer.writerow({"prompt_id": entry["id"], "block": entry["block"]})

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"  prompts in bank: {len(bank)}")
    logger.info(f"  annotations.csv initialized with {len(bank)} empty rows.")

    if args.dry_run:
        logger.info("--dry-run: exiting without loading model.")
        return 0

    sessions = group_into_sessions(bank)
    logger.info(
        f"  grouped into {len(sessions)} session(s) "
        f"({sum(1 for s in sessions if len(s) > 1)} multi-turn)."
    )

    runner = load_model(cfg)
    for session in sessions:
        wav_paths: list[Path] = []
        out_dirs: list[Path] = []
        skipped: list[str] = []
        for entry in session:
            wav_path = args.wav / f"{entry['id']}.wav"
            if not wav_path.exists():
                skipped.append(entry["id"])
                continue
            out_dir = run_dir / "outputs" / entry["id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            wav_paths.append(wav_path)
            out_dirs.append(out_dir)
        if skipped:
            logger.warning(f"  [skip session] missing wavs: {skipped} (run synth_prompts first).")
            continue
        if not wav_paths:
            continue

        session_label = (
            session[0].get("sequence")
            or (session[0]["id"] if len(session) == 1 else f"{session[0]['id']}..{session[-1]['id']}")
        )
        outputs = runner.run_session(wav_paths, out_dirs)
        for entry, output, out_dir in zip(session, outputs, out_dirs):
            (out_dir / "output.json").write_text(json.dumps(asdict(output), indent=2))
            logger.info(
                f"  [{session_label}/{entry['id']}] "
                f"{output.n_text_tokens} text tok / {output.elapsed_sec:.1f}s"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
