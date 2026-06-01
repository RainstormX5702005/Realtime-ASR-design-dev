from __future__ import annotations

import numpy as np

from audio_queue import AudioTask
from transcription.base import (
    BaseTranscriber,
    RawTranscriptionResult,
    TranscriptionResult,
)


class DummyTranscriber(BaseTranscriber):
    def transcribe(self, task: AudioTask) -> TranscriptionResult:
        raw = RawTranscriptionResult(
            text="hello",
            nbest_tokens=[[1, 2], [3]],
            nbest_scores=[-1.0, -2.0],
            model_name="dummy",
        )
        return TranscriptionResult(
            segment_id=task.segment_id,
            window_index=task.window_index,
            text=raw.text,
            duration_ms=100.0,
            is_final_window=task.is_final_window,
            raw=raw,
        )

    def close(self) -> None:
        return None


def test_backend_contract_preserves_raw_nbest_and_task_metadata():
    task = AudioTask(
        segment_id=7,
        window_index=2,
        audio=np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        is_final_window=True,
        created_at=1.0,
    )

    result = DummyTranscriber().transcribe(task)

    assert result.segment_id == 7
    assert result.window_index == 2
    assert result.text == "hello"
    assert result.is_final_window is True
    assert result.error is None
    assert result.raw is not None
    assert result.raw.model_name == "dummy"
    assert result.raw.nbest_tokens == [[1, 2], [3]]
    assert result.raw.nbest_scores == [-1.0, -2.0]
