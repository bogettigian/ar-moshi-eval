from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class PromptOutput:
    duration_sec: float
    n_steps: int
    n_text_tokens: int
    transcript: str
    elapsed_sec: float
    tokens_per_sec: float


@runtime_checkable
class ModelRunner(Protocol):
    def run_session(
        self, wav_paths: list[Path], out_dirs: list[Path]
    ) -> list[PromptOutput]: ...