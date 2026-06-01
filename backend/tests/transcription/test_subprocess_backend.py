from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from audio_queue import AudioTask
from transcription.subprocess_backend import (
    SubprocessTranscriber,
    SubprocessTranscriberConfig,
)


def _write_child_script(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _task() -> AudioTask:
    return AudioTask(
        segment_id=3,
        window_index=1,
        audio=np.zeros(800, dtype=np.float32),
        sample_rate=16000,
        is_final_window=True,
        created_at=1.0,
    )


def test_subprocess_transcriber_returns_result_and_deletes_temp_wav(tmp_path):
    seen_path = tmp_path / "seen.json"
    child = tmp_path / "child.py"
    _write_child_script(
        child,
        f"""
import json
import sys
from pathlib import Path

print(json.dumps({{"status": "ready", "model_name": "fake"}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "shutdown":
        print(json.dumps({{"status": "bye"}}), flush=True)
        break
    Path({str(seen_path)!r}).write_text(
        json.dumps({{"wav": request["wav"]}}), encoding="utf-8"
    )
    print(json.dumps({{
        "status": "ok",
        "text": "hello",
        "model_name": "fake",
        "nbest_tokens": [[1, 2], [3]],
        "nbest_scores": [-0.1, -0.2],
    }}), flush=True)
""",
    )
    transcriber = SubprocessTranscriber(
        SubprocessTranscriberConfig(command=[sys.executable, str(child)])
    )

    result = transcriber.transcribe(_task())
    transcriber.close()

    request = json.loads(seen_path.read_text(encoding="utf-8"))
    assert result.error is None
    assert result.text == "hello"
    assert result.segment_id == 3
    assert result.window_index == 1
    assert result.duration_ms == 50.0
    assert result.raw is not None
    assert result.raw.model_name == "fake"
    assert result.raw.nbest_tokens == [[1, 2], [3]]
    assert result.raw.nbest_scores == [-0.1, -0.2]
    assert not Path(request["wav"]).exists()


def test_subprocess_transcriber_maps_child_error_to_result_error(tmp_path):
    child = tmp_path / "child.py"
    _write_child_script(
        child,
        """
import json
import sys

print(json.dumps({"status": "ready", "model_name": "fake"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "shutdown":
        break
    print(
        json.dumps({"status": "error", "error": "decode failed"}),
        flush=True,
    )
""",
    )
    transcriber = SubprocessTranscriber(
        SubprocessTranscriberConfig(command=[sys.executable, str(child)])
    )

    result = transcriber.transcribe(_task())
    transcriber.close()

    assert result.text == ""
    assert result.error == "decode failed"
    assert result.raw is None


def test_subprocess_transcriber_sends_shutdown_on_close(tmp_path):
    marker = tmp_path / "shutdown.txt"
    child = tmp_path / "child.py"
    _write_child_script(
        child,
        f"""
import json
import sys
from pathlib import Path

print(json.dumps({{"status": "ready", "model_name": "fake"}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "shutdown":
        Path({str(marker)!r}).write_text("closed", encoding="utf-8")
        break
""",
    )
    transcriber = SubprocessTranscriber(
        SubprocessTranscriberConfig(command=[sys.executable, str(child)])
    )

    transcriber.close()

    assert marker.read_text(encoding="utf-8") == "closed"


def test_subprocess_transcriber_times_out_waiting_for_ready(tmp_path):
    child = tmp_path / "child.py"
    _write_child_script(
        child,
        """
import time

time.sleep(10)
""",
    )

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="ready"):
        SubprocessTranscriber(
            SubprocessTranscriberConfig(
                command=[sys.executable, str(child)],
                ready_timeout_s=0.1,
            )
        )

    assert time.monotonic() - start < 2.0


def test_subprocess_transcriber_cleans_up_after_bad_ready_response(tmp_path):
    child = tmp_path / "child.py"
    marker = tmp_path / "terminated.txt"
    _write_child_script(
        child,
        f"""
import json
import signal
import sys
import time
from pathlib import Path

def handle_term(signum, frame):
    Path({str(marker)!r}).write_text("terminated", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_term)
print(json.dumps({{"status": "not-ready"}}), flush=True)
while True:
    time.sleep(1)
""",
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        SubprocessTranscriber(
            SubprocessTranscriberConfig(command=[sys.executable, str(child)])
        )

    assert marker.read_text(encoding="utf-8") == "terminated"
