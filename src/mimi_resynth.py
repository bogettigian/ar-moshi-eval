from __future__ import annotations

import argparse
import csv
import logging
import logging.config
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import sphn
import torch
from moshi.models import loaders
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.auto_annotate import compute_ppl, pick_device_and_dtype, transcribe

logger = logging.getLogger(__name__)

RESYNTH_COLUMNS = [
    "file",
    "channel",
    "seconds",
    "transcript_original",
    "transcript_resynth",
    "ppl_original",
    "ppl_resynth",
    "ppl_delta",
    "wer",
    "asr_model",
    "lm_model",
    "mimi_checkpoint",
]

PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> list[str]:
    return PUNCTUATION.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return float("nan")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            substitution = previous[j - 1] + (ref_word != hyp_word)
            current.append(min(substitution, previous[j] + 1, current[j - 1] + 1))
        previous = current
    return previous[-1] / len(ref)


def collect_audio(paths: list[Path], limit: int | None) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.wav")))
        else:
            files.append(path)
    if limit is not None:
        files = files[:limit]
    return files


def resynthesize(mimi, pcm: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(pcm)[None, None].to(device)
    with torch.no_grad():
        codes = mimi.encode(x)
        out = mimi.decode(codes)
    return out[0, 0].cpu().numpy()


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Measure Mimi re-synthesis quality on Spanish audio.")
    parser.add_argument("audio", nargs="+", type=Path, help="Wav files, or directories of wavs.")
    parser.add_argument("--out-dir", type=Path, default=Path("./reports/mimi_resynth"), help="Where the original/reconstructed chunks and the CSV are written.")
    parser.add_argument("--seconds", type=float, default=120.0, help="Seconds taken from each file. Default: 120.")
    parser.add_argument("--start-sec", type=float, default=60.0, help="Offset into each file, to skip intros and jingles. Default: 60.")
    parser.add_argument("--channel", type=int, default=0, help="Which channel of the stereo corpus to measure. Default: 0.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many files.")
    parser.add_argument("--checkpoint", default=loaders.DEFAULT_REPO, help=f"HF repo id the Mimi weights come from. Default: {loaders.DEFAULT_REPO}.")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3", help="HF repo id of the ASR model. Default: openai/whisper-large-v3.")
    parser.add_argument("--lm-model", default="BSC-LT/salamandra-2b", help="HF repo id of the causal LM used to compute PPL. Default: BSC-LT/salamandra-2b.")
    parser.add_argument("--device", default=None, help="Override the auto-detected device.")
    args = parser.parse_args()

    files = collect_audio(args.audio, args.limit)
    if not files:
        logger.error("No audio files found.")
        return 1

    auto_device, dtype = pick_device_and_dtype()
    device = args.device or auto_device
    logger.info(f"Device: {device} (dtype={dtype})")
    logger.info(f"{len(files)} file(s), {args.seconds:.0f}s each from {args.start_sec:.0f}s in.")

    logger.info(f"Loading Mimi from: {args.checkpoint}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.checkpoint)
    mimi = checkpoint_info.get_mimi(device=device)
    sample_rate = int(mimi.sample_rate)

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

    wav_dir = args.out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        arr, _ = sphn.read(
            str(path),
            sample_rate=sample_rate,
            start_sec=args.start_sec,
            duration_sec=args.seconds,
        )
        if args.channel >= arr.shape[0]:
            logger.warning(f"  [{i}/{len(files)}] {path.name} has {arr.shape[0]} channel(s), skipping channel {args.channel}.")
            continue
        original = np.ascontiguousarray(arr[args.channel])
        resynth = resynthesize(mimi, original, device)

        original_path = wav_dir / f"{path.stem}_ch{args.channel}_original.wav"
        resynth_path = wav_dir / f"{path.stem}_ch{args.channel}_resynth.wav"
        sphn.write_wav(str(original_path), original, sample_rate=sample_rate)
        sphn.write_wav(str(resynth_path), resynth, sample_rate=sample_rate)

        transcript_original = transcribe(asr_pipe, original_path)
        transcript_resynth = transcribe(asr_pipe, resynth_path)
        ppl_original = compute_ppl(lm_tokenizer, lm_model, transcript_original)
        ppl_resynth = compute_ppl(lm_tokenizer, lm_model, transcript_resynth)
        wer = word_error_rate(transcript_original, transcript_resynth)

        logger.info(
            f"  [{i}/{len(files)}] {path.name} — "
            f"ppl {ppl_original:.1f} → {ppl_resynth:.1f}, wer={wer:.3f}"
        )
        rows.append(
            {
                "file": path.name,
                "channel": args.channel,
                "seconds": f"{original.shape[-1] / sample_rate:.1f}",
                "transcript_original": transcript_original,
                "transcript_resynth": transcript_resynth,
                "ppl_original": "" if math.isnan(ppl_original) else f"{ppl_original:.4f}",
                "ppl_resynth": "" if math.isnan(ppl_resynth) else f"{ppl_resynth:.4f}",
                "ppl_delta": "" if math.isnan(ppl_original) or math.isnan(ppl_resynth) else f"{ppl_resynth - ppl_original:.4f}",
                "wer": "" if math.isnan(wer) else f"{wer:.4f}",
                "asr_model": args.asr_model,
                "lm_model": args.lm_model,
                "mimi_checkpoint": args.checkpoint,
            }
        )

    if not rows:
        logger.error("Nothing was measured.")
        return 1

    csv_path = args.out_dir / "mimi_resynth.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESYNTH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {csv_path} with {len(rows)} row(s) in {time.time() - t0:.1f}s.")

    def mean(column: str) -> float:
        values = [float(r[column]) for r in rows if r[column] != ""]
        return sum(values) / len(values) if values else float("nan")

    print(f"files measured:  {len(rows)}")
    print(f"mean PPL before: {mean('ppl_original'):.1f}")
    print(f"mean PPL after:  {mean('ppl_resynth'):.1f}")
    print(f"mean WER:        {mean('wer'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
