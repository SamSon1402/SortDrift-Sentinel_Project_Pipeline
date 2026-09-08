from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


def main(path: str, endpoint: str = "http://127.0.0.1:8010/v1/events") -> None:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"file not found: {source}")
    accepted = 0
    with source.open("r", encoding="utf-8") as handle, httpx.Client(timeout=10) as client:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            accepted += 1
    print(f"replayed {accepted} events -> {endpoint}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/replay_from_garment_grader.py events.jsonl")
    main(sys.argv[1])
