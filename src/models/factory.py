from __future__ import annotations

from src.models.base import ModelRunner
from src.models.moshi_pytorch import MoshiPytorchRunner


def load_model(cfg) -> ModelRunner:
    if cfg.model_type == "moshi_pytorch":
        return MoshiPytorchRunner.load(cfg)
    raise ValueError(f"Unknown model_type: {cfg.model_type!r}")