#!/usr/bin/env python3
"""Wenet JSON Lines subprocess service."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from typing import Any

import torch
from wenet.cli.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Wenet over stdin/stdout JSONL"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=10)
    return parser.parse_args()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def transcribe_wav(model, wav_path: str, beam_size: int) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        with redirect_stdout(sys.stderr):
            speech = model.compute_feature(wav_path).to(device)
            speech_lengths = torch.tensor([speech.size(0)], device=device)
            speech = speech.unsqueeze(0)
            methods = ["ctc_prefix_beam_search"]
            default_method = getattr(
                model, "default_decode_method", "ctc_prefix_beam_search"
            )
            if default_method not in methods:
                methods.append(default_method)
            results = model.decode(
                methods, speech, speech_lengths, beam_size=beam_size
            )

    prefix_result = results["ctc_prefix_beam_search"][0]
    best_result = results.get(
        default_method, results["ctc_prefix_beam_search"]
    )[0]
    text = model.tokenizer.detokenize(best_result.tokens)[0]
    nbest_tokens = [list(tokens) for tokens in (prefix_result.nbest or [])]
    nbest_scores = [
        float(score) for score in (prefix_result.nbest_scores or [])
    ]
    return {
        "status": "ok",
        "text": text,
        "model_name": "wenet",
        "nbest_tokens": nbest_tokens,
        "nbest_scores": nbest_scores,
    }


def handle_request(
    model, request: dict[str, Any], beam_size: int
) -> dict[str, Any] | None:
    if request.get("command") == "shutdown":
        return None
    wav_path = request.get("wav")
    if not wav_path:
        return {"status": "error", "error": "missing wav"}
    try:
        return transcribe_wav(model, str(wav_path), beam_size)
    except Exception as exc:  # noqa: BLE001
        print(f"Wenet request failed: {exc}", file=sys.stderr, flush=True)
        return {"status": "error", "error": str(exc)}


def main() -> int:
    args = parse_args()
    print("Loading Wenet model...", file=sys.stderr, flush=True)
    with redirect_stdout(sys.stderr):
        model = load_model(args.model_dir, device=args.device)
    emit({"status": "ready", "model_name": "wenet"})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"status": "error", "error": f"invalid json: {exc}"})
            continue
        response = handle_request(model, request, args.beam_size)
        if response is None:
            break
        emit(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
