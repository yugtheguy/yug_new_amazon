#!/usr/bin/env python3
"""Merge independently scored country artifacts through the frozen decision layer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/final_test"))
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/final_b007"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_submission"))
    parser.add_argument("--model-id", choices=("B007_03", "B007_01"), default="B007_03")
    parser.add_argument("--country", action="append", required=True)
    parser.add_argument("--train-source2", type=Path); parser.add_argument("--train-source3", type=Path)
    args = parser.parse_args()
    predictions = []
    for country in args.country:
        path = args.artifacts_dir / country / "predictions.parquet"
        done = args.artifacts_dir / country / "DONE"
        if not path.exists() or not done.exists(): raise FileNotFoundError(f"Incomplete country checkpoint: {country}")
        predictions.append(path)
    command = [sys.executable, str(ROOT / "scripts/apply_decision_rule.py"), "--predictions", *map(str, predictions), "--test-dir", str(args.test_dir), "--model-dir", str(args.model_dir), "--output-dir", str(args.output_dir), "--model-id", args.model_id]
    if args.train_source2 and args.train_source3: command += ["--train-source2", str(args.train_source2), "--train-source3", str(args.train_source3)]
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
