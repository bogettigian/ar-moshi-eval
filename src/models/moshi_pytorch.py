from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sphn
import torch
from moshi.conditioners import ConditionAttributes
from moshi.models import LMGen, loaders

from src.models.base import PromptOutput


NO_TOKEN_IDS = (0, 3)


@dataclass
class MoshiPytorchRunner:
    mimi: object # MimiModel
    text_tokenizer: object # sentencepiece.SentencePieceProcessor
    lm_gen: object # LMGen
    device: str
    sample_rate: int
    frame_size: int
    dep_q: int
    temperature_audio: float
    temperature_text: float
    top_k_audio: int
    top_k_text: int
    cfg_coef: float

    @classmethod
    def load(cls, cfg) -> "MoshiPytorchRunner":
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        device = cfg.device
        dtype_str = getattr(cfg, "dtype", None) or ("float16" if device == "mps" else "bfloat16")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]

        ckpt = cfg.checkpoint
        lora_weights = getattr(cfg, "lora_weight", None)
        if Path(ckpt).is_dir():
            ckpt_dir = Path(ckpt)
            config_path = ckpt_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Local checkpoint dir {ckpt_dir} has no config.json. "
                    "Expected layout: <dir>/config.json + weight safetensors."
                )
            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
                loaders.DEFAULT_REPO,
                config_path=str(config_path),
                lora_weights=lora_weights,
            )
        else:
            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
                ckpt, lora_weights=lora_weights,
            )
        mimi = checkpoint_info.get_mimi(device=device)
        text_tokenizer = checkpoint_info.get_text_tokenizer()
        lm = checkpoint_info.get_moshi(device=device, dtype=dtype)

        condition_tensors = {}
        if lm.condition_provider is not None and lm.condition_provider.conditioners:
            conditions = [ConditionAttributes(text={"description": "very_good"}, tensor={})]
            if cfg.cfg_coef != 1.0:
                conditions.append(ConditionAttributes(text={"description": "very_bad"}, tensor={}))
            condition_tensors = lm.condition_provider.prepare_and_provide(conditions)

        lm_gen = LMGen(
            lm,
            temp=cfg.temperature_audio,
            temp_text=cfg.temperature_text,
            top_k=cfg.top_k_audio,
            top_k_text=cfg.top_k_text,
            cfg_coef=cfg.cfg_coef,
            condition_tensors=condition_tensors,
            **checkpoint_info.lm_gen_config,
        )

        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        return cls(
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm_gen=lm_gen,
            device=device,
            sample_rate=int(mimi.sample_rate),
            frame_size=int(mimi.sample_rate / mimi.frame_rate),
            dep_q=lm.dep_q,
            temperature_audio=cfg.temperature_audio,
            temperature_text=cfg.temperature_text,
            top_k_audio=cfg.top_k_audio,
            top_k_text=cfg.top_k_text,
            cfg_coef=cfg.cfg_coef,
        )

    def run_session(
        self, wav_paths: list[Path], out_dirs: list[Path]
    ) -> list[PromptOutput]:
        if len(wav_paths) != len(out_dirs):
            raise ValueError(
                f"wav_paths ({len(wav_paths)}) and out_dirs ({len(out_dirs)}) length mismatch"
            )

        # Reset streaming state once at session start. Within a session, state
        # carries over across turns — that's how the model recalls earlier turns.
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

        pcms: list[torch.Tensor] = []
        for p in wav_paths:
            arr, _ = sphn.read(str(p), sample_rate=self.sample_rate)
            t = torch.from_numpy(arr).to(device=self.device)
            t = t[None, 0:1]  # (1, 1, T) — batch=1, channel=1
            pcms.append(t)

        outputs: list[PromptOutput] = []
        first_frame_of_session = True
        global_step = 0

        with torch.no_grad():
            for pcm, out_dir in zip(pcms, out_dirs):
                n_samples = pcm.shape[-1]
                n_steps = n_samples // self.frame_size

                text_pieces: list[str] = []
                text_tokens_log: list[tuple[int, int]] = []
                all_out_pcm: list[np.ndarray] = []

                start = time.time()
                for idx in range(n_steps):
                    chunk = pcm[:, :, idx * self.frame_size : (idx + 1) * self.frame_size]
                    codes = self.mimi.encode(chunk)

                    if first_frame_of_session:
                        _ = self.lm_gen.step(codes)
                        first_frame_of_session = False

                    tokens = self.lm_gen.step(codes)
                    if tokens is None:
                        text_tokens_log.append((global_step, 0))
                        global_step += 1
                        continue

                    text_token = int(tokens[0, 0, 0].cpu().item())
                    text_tokens_log.append((global_step, text_token))
                    if text_token not in NO_TOKEN_IDS:
                        piece = self.text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                        text_pieces.append(piece)

                    if self.dep_q > 0:
                        out_pcm = self.mimi.decode(tokens[:, 1:]).cpu().numpy()
                        all_out_pcm.append(out_pcm)
                    global_step += 1
                elapsed = time.time() - start

                out_wav_path = out_dir / "response.wav"
                if all_out_pcm:
                    joined = np.concatenate(all_out_pcm, axis=-1)
                    sphn.write_wav(str(out_wav_path), joined[0, 0], sample_rate=self.sample_rate)
                else:
                    sphn.write_wav(str(out_wav_path), np.zeros(1, dtype=np.float32), sample_rate=self.sample_rate)

                with (out_dir / "text_tokens.txt").open("w") as f:
                    for step_idx, tok_id in text_tokens_log:
                        piece = self.text_tokenizer.id_to_piece(tok_id) if tok_id not in NO_TOKEN_IDS else ""
                        f.write(f"{step_idx}\t{tok_id}\t{piece}\n")

                outputs.append(
                    PromptOutput(
                        duration_sec=n_samples / float(self.sample_rate),
                        n_steps=n_steps,
                        n_text_tokens=sum(1 for _, t in text_tokens_log if t not in NO_TOKEN_IDS),
                        transcript="".join(text_pieces).strip(),
                        elapsed_sec=elapsed,
                        tokens_per_sec=n_steps / elapsed if elapsed > 0 else 0.0,
                    )
                )

        return outputs