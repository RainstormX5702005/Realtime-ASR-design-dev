"""Shared transcription backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from audio_queue import AudioTask


@dataclass(frozen=True)
class RawTranscriptionResult:
    """Backend-native ASR output, including complete n-best data."""

    text: str
    nbest_tokens: list[list[int]]
    nbest_scores: list[float]
    model_name: str


@dataclass(frozen=True)
class TranscriptionResult:
    """ASR result enriched with audio task metadata."""

    segment_id: int
    window_index: int
    text: str
    duration_ms: float
    is_final_window: bool
    raw: RawTranscriptionResult | None = None
    error: str | None = None


class BaseTranscriber(ABC):
    """Common interface implemented by all ASR backends."""

    @abstractmethod
    def transcribe(self, task: AudioTask) -> TranscriptionResult:
        """Transcribes one queued audio task."""

    @abstractmethod
    def close(self) -> None:
        """Releases backend resources."""
