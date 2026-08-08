import argparse
import logging
import logging.config
import sys
from pathlib import Path

from TTS.api import TTS

import numpy as np
import soundfile as sf
import yaml

logger = logging.getLogger(__name__)


def append_silence(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    if seconds <= 0:
        return wav
    silence = np.zeros(int(sr * seconds), dtype=wav.dtype)
    return np.concatenate([wav, silence])


def synthesize_one(tts, entry: dict, out_path: Path, speaker: str | None, speaker_wav: str | None) -> None:
    tmp_path = out_path.with_suffix(".tmp.wav")
    kwargs = dict(text=entry["text"], language="es", file_path=str(tmp_path))
    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav
    else:
        kwargs["speaker"] = speaker
    tts.tts_to_file(**kwargs)

    wav, sr = sf.read(tmp_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 24_000:
        raise RuntimeError(f"Expected 24 kHz from XTTS, got {sr} for {entry['id']}")
    wav = append_silence(wav.astype(np.float32), sr, float(entry.get("trailing_silence_sec", 10)))
    sf.write(out_path, wav, sr, subtype="PCM_16")
    tmp_path.unlink(missing_ok=True)


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description="Synthesize prompts/bank.yaml to wav with XTTS v2.")
    parser.add_argument("--force", action="store_true", help="Re-synthesize even if the wav already exists.")
    parser.add_argument("--speaker", default="Claribel Dervla", help="Built-in XTTS v2 speaker.")
    parser.add_argument("--voice-sample", default=None, help="Path to a reference wav for voice cloning (optional).")
    parser.add_argument("--only", default=None, help="Single ID to synthesize (debug).")
    parser.add_argument("--bank", type=Path, default=Path("./prompts/bank.yaml"), help="Path to the prompt bank.")
    parser.add_argument("--wav", type=Path, default=Path("./prompts/wav"), help="Path to the wav output dir.")
    args = parser.parse_args()

    with args.bank.open() as f:
        bank = yaml.safe_load(f)

    args.wav.mkdir(parents=True, exist_ok=True)

    pending = [
        e for e in bank
        if (args.only is None or e["id"] == args.only)
        and (args.force or not (args.wav / f"{e['id']}.wav").exists())
    ]
    if not pending:
        logger.info("Nothing to synthesize (use --force to regenerate).")
        return 0

    logger.info(f"Loading XTTS v2 ({len(pending)} pending prompts)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    for entry in pending:
        out_path = args.wav / f"{entry['id']}.wav"
        logger.info(f"  [{entry['id']}] {entry['text'][:60]}")
        synthesize_one(tts, entry, out_path, args.speaker, args.voice_sample)

    logger.info(f"Done. {len(pending)} wav(s) in {args.wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())