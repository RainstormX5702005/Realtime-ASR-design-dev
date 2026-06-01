"""Factory for ASR transcriber backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcription.base import BaseTranscriber
from transcription.subprocess_backend import (
    SubprocessTranscriber,
    SubprocessTranscriberConfig,
)


@dataclass(frozen=True)
class TranscriberFactoryConfig:
    """Configuration used to create an ASR backend."""

    backend: str = "wenet"
    model_dir: Path = Path("models")
    device: str = "cuda"
    beam_size: int = 10
    script_path: Path = Path("scripts/wenet_serve.sh")


def create_transcriber(config: TranscriberFactoryConfig) -> BaseTranscriber:
    """Creates the configured ASR backend."""

    if config.backend != "wenet":
        raise ValueError(f"Unsupported ASR backend: {config.backend}")

    command = [
        str(config.script_path),
        "--model-dir",
        str(config.model_dir),
        "--device",
        config.device,
        "--beam-size",
        str(config.beam_size),
    ]
    return SubprocessTranscriber(
        SubprocessTranscriberConfig(command=command, model_name=config.backend)
    )
