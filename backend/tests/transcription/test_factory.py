from __future__ import annotations

import pytest

from transcription.factory import TranscriberFactoryConfig, create_transcriber
from transcription.subprocess_backend import SubprocessTranscriber


def test_create_wenet_transcriber_builds_subprocess_command(
    monkeypatch,
    tmp_path,
):
    created_configs = []

    class FakeSubprocessTranscriber:
        def __init__(self, config):
            created_configs.append(config)

    monkeypatch.setattr(
        "transcription.factory.SubprocessTranscriber",
        FakeSubprocessTranscriber,
    )
    script_path = tmp_path / "wenet_serve.sh"
    model_dir = tmp_path / "model"

    transcriber = create_transcriber(
        TranscriberFactoryConfig(
            backend="wenet",
            model_dir=model_dir,
            device="cpu",
            beam_size=8,
            script_path=script_path,
        )
    )

    assert isinstance(transcriber, FakeSubprocessTranscriber)
    config = created_configs[0]
    assert config.model_name == "wenet"
    assert config.command == [
        str(script_path),
        "--model-dir",
        str(model_dir),
        "--device",
        "cpu",
        "--beam-size",
        "8",
    ]


def test_create_transcriber_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unsupported ASR backend"):
        create_transcriber(
            TranscriberFactoryConfig(
                backend="unknown",
                model_dir=tmp_path / "model",
            )
        )


def test_transcription_package_exports_new_backend_api():
    from transcription import (
        BaseTranscriber,
        TranscriptionResult,
        create_transcriber,
    )

    assert BaseTranscriber is not None
    assert TranscriptionResult is not None
    assert create_transcriber is not None
    assert SubprocessTranscriber is not None
