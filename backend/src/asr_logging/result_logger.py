"""Async JSONL logger for ASR results and service events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from transcription.base import TranscriptionResult


@dataclass(frozen=True)
class AsrLogConfig:
    """Configuration for ASR JSONL logs."""

    log_dir: Path
    enabled: bool = True


class AsrResultLogger:
    """Writes ASR results and service events to a daily JSONL file."""

    def __init__(self, config: AsrLogConfig):
        self.config = config
        self._queue: asyncio.Queue[dict[str, Any] | None] | None = None
        self._writer_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self.config.enabled or self._writer_task is not None:
            return
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._writer())

    async def stop(self) -> None:
        if self._queue is None or self._writer_task is None:
            return
        await self._queue.put(None)
        await self._writer_task
        self._queue = None
        self._writer_task = None

    def record_result(self, result: TranscriptionResult) -> None:
        raw = result.raw
        self._record(
            {
                "type": "result",
                "timestamp": self._timestamp(),
                "segment_id": result.segment_id,
                "window_index": result.window_index,
                "duration_ms": result.duration_ms,
                "is_final_window": result.is_final_window,
                "model_name": raw.model_name if raw is not None else "",
                "text": result.text,
                "nbest_tokens": raw.nbest_tokens if raw is not None else [],
                "nbest_scores": raw.nbest_scores if raw is not None else [],
                "error": result.error,
            }
        )

    def record_event(self, event: str, **fields: Any) -> None:
        payload = {
            "type": "event",
            "timestamp": self._timestamp(),
            "event": event,
        }
        payload.update(fields)
        self._record(payload)

    def _record(self, payload: dict[str, Any]) -> None:
        if not self.config.enabled or self._queue is None:
            return
        self._queue.put_nowait(payload)

    async def _writer(self) -> None:
        assert self._queue is not None
        date_tag = datetime.now().strftime("%Y%m%d")
        log_path = self.config.log_dir / f"transcription_{date_tag}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            while True:
                payload = await self._queue.get()
                if payload is None:
                    self._queue.task_done()
                    break
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f.flush()
                self._queue.task_done()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
