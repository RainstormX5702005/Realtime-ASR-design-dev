from __future__ import annotations

import asyncio
import json

from asr_logging import AsrLogConfig, AsrResultLogger
from transcription.base import RawTranscriptionResult, TranscriptionResult


def test_result_logger_writes_result_and_event_rows(tmp_path):
    async def run_test():
        logger = AsrResultLogger(AsrLogConfig(log_dir=tmp_path))
        await logger.start()
        logger.record_result(
            TranscriptionResult(
                segment_id=5,
                window_index=0,
                text="hello",
                duration_ms=250.0,
                is_final_window=True,
                raw=RawTranscriptionResult(
                    text="hello",
                    nbest_tokens=[[1, 2], [3]],
                    nbest_scores=[-0.1, -0.2],
                    model_name="fake",
                ),
            )
        )
        logger.record_event("subprocess_ready", backend="fake")
        await logger.stop()

    asyncio.run(run_test())

    log_files = list(tmp_path.glob("transcription_*.jsonl"))
    assert len(log_files) == 1
    rows = [json.loads(line) for line in log_files[0].read_text().splitlines()]
    assert rows[0]["type"] == "result"
    assert rows[0]["segment_id"] == 5
    assert rows[0]["window_index"] == 0
    assert rows[0]["model_name"] == "fake"
    assert rows[0]["text"] == "hello"
    assert rows[0]["nbest_tokens"] == [[1, 2], [3]]
    assert rows[0]["nbest_scores"] == [-0.1, -0.2]
    assert rows[0]["error"] is None
    assert rows[1]["type"] == "event"
    assert rows[1]["event"] == "subprocess_ready"
    assert rows[1]["backend"] == "fake"


def test_result_logger_records_error_result_without_raw(tmp_path):
    async def run_test():
        logger = AsrResultLogger(AsrLogConfig(log_dir=tmp_path))
        await logger.start()
        logger.record_result(
            TranscriptionResult(
                segment_id=1,
                window_index=2,
                text="",
                duration_ms=50.0,
                is_final_window=False,
                error="decode failed",
            )
        )
        await logger.stop()

    asyncio.run(run_test())

    log_file = next(tmp_path.glob("transcription_*.jsonl"))
    row = json.loads(log_file.read_text().strip())
    assert row["type"] == "result"
    assert row["model_name"] == ""
    assert row["nbest_tokens"] == []
    assert row["nbest_scores"] == []
    assert row["error"] == "decode failed"
