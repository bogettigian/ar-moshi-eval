from __future__ import annotations

import argparse
import csv
import logging
import logging.config
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

AUTO_METRICS_COLUMNS = [
    "prompt_id",
    "block",
    "transcript_asr",
    "ppl",
    "lang_detected",
    "lang_confidence",
    "asr_model",
    "lm_model",
    "lang_id_model",
]

# Models whose identity invalidates a cached row: if any of them changed since
# the row was written, the row has to be recomputed.
MODEL_COLUMNS = ("asr_model", "lm_model", "lang_id_model")


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def transcribe(asr_pipe, wav_path: Path) -> str:
    out = asr_pipe(str(wav_path))
    text = out.get("text") if isinstance(out, dict) else None
    return (text or "").strip()


def detect_language(lang_pipe, text: str) -> tuple[str, float]:
    if not text:
        return "", float("nan")
    out = lang_pipe(text, truncation=True, max_length=512)
    if isinstance(out, list):
        out = out[0]
    if isinstance(out, list):  # top_k > 1 returns a list per input
        out = out[0]
    return str(out["label"]), float(out["score"])


def compute_ppl(lm_tokenizer, lm_model, text: str) -> float:
    if not text:
        return float("nan")
    enc = lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc.input_ids.to(lm_model.device)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = lm_model(input_ids=input_ids, labels=input_ids)
    return math.exp(float(out.loss))


def load_existing(metrics_path: Path) -> dict[str, dict]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as f:
        return {row["prompt_id"]: row for row in csv.DictReader(f)}


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Auto-annotate a run with ASR transcript + LM perplexity.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to runs/<id>/.")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3", help=f"HF repo id of the ASR model. Default: openai/whisper-large-v3.")
    parser.add_argument("--lm-model", default="BSC-LT/salamandra-2b", help=f"HF repo id of the causal LM used to compute PPL. Default: BSC-LT/salamandra-2b.")
    parser.add_argument("--lang-id-model", default="papluca/xlm-roberta-base-language-detection", help="HF repo id of the text classifier used to label the transcript's language. Default: papluca/xlm-roberta-base-language-detection.")
    parser.add_argument("--bank", type=Path, default=Path("./prompts/bank.yaml"), help="Path to the prompt bank.")
    parser.add_argument("--force", action="store_true", help="Recompute every row even if auto_metrics.csv already has it.")
    args = parser.parse_args()

    outputs_dir = args.run_dir / "outputs"
    if not outputs_dir.is_dir():
        logger.error(f"{outputs_dir} does not exist or is not a directory.")
        return 1

    with args.bank.open() as f:
        bank = yaml.safe_load(f)
    block_of = {entry["id"]: entry["block"] for entry in bank}

    metrics_path = args.run_dir / "auto_metrics.csv"
    existing = {} if args.force else load_existing(metrics_path)

    todo: list[str] = []
    for entry in bank:
        pid = entry["id"]
        if not (outputs_dir / pid / "response.wav").exists():
            continue
        row = existing.get(pid)
        requested = (args.asr_model, args.lm_model, args.lang_id_model)
        if row and tuple(row.get(col) for col in MODEL_COLUMNS) == requested:
            continue
        todo.append(pid)

    if not todo:
        logger.info(f"Nothing to do. {metrics_path} already has all rows with the requested models.")
        return 0

    device, dtype = pick_device_and_dtype()
    logger.info(f"Device: {device} (dtype={dtype})")

    logger.info(f"Loading ASR: {args.asr_model}")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=device,
        torch_dtype=dtype,
    )

    logger.info(f"Loading LM:  {args.lm_model}")
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model, torch_dtype=dtype)
    lm_model.to(device)
    lm_model.eval()

    logger.info(f"Loading language ID: {args.lang_id_model}")
    lang_pipe = pipeline(
        "text-classification",
        model=args.lang_id_model,
        device=device,
        torch_dtype=dtype,
    )

    rows: dict[str, dict] = {} if args.force else dict(existing)
    t0 = time.time()
    for i, pid in enumerate(todo, 1):
        wav_path = outputs_dir / pid / "response.wav"
        transcript = transcribe(asr_pipe, wav_path)
        ppl = compute_ppl(lm_tokenizer, lm_model, transcript)
        lang, lang_conf = detect_language(lang_pipe, transcript)
        ppl_str = f"ppl={ppl:.1f}" if not math.isnan(ppl) else "ppl=nan"
        lang_str = f"lang={lang} ({lang_conf:.2f})" if lang else "lang=none"
        logger.info(f"  [{i}/{len(todo)}] {pid} — {ppl_str} {lang_str}")
        rows[pid] = {
            "prompt_id": pid,
            "block": block_of.get(pid, ""),
            "transcript_asr": transcript,
            "ppl": "" if math.isnan(ppl) else f"{ppl:.4f}",
            "lang_detected": lang,
            "lang_confidence": "" if math.isnan(lang_conf) else f"{lang_conf:.4f}",
            "asr_model": args.asr_model,
            "lm_model": args.lm_model,
            "lang_id_model": args.lang_id_model,
        }

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUTO_METRICS_COLUMNS)
        writer.writeheader()
        for entry in bank:
            pid = entry["id"]
            if pid in rows:
                writer.writerow(rows[pid])

    n_written = sum(1 for e in bank if e["id"] in rows)
    logger.info(f"Wrote {metrics_path} with {n_written} row(s) in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
