#!/usr/bin/env python3
"""Wrap run_generation_private.py and capture its stdout metrics into a JSON summary.

The upstream GPT-2 private generation script prints a single result line from
rank 0 but writes no file. This wrapper runs it as a subprocess, parses the
"private generation step: ..." line, and writes <out-dir>/summary.json plus
stdout.log and replay_status.json so the result tree matches the e2e suites.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root  # noqa: E402

SHAFT_ROOT = get_shaft_root()
GPT2_SCRIPT = SHAFT_ROOT / "examples" / "text-generation" / "run_generation_private.py"
# GPT-2 weights: override via $GPT2_MODEL, otherwise let HF download to the cache.
GPT2_MODEL = Path(os.environ["GPT2_MODEL"]) if os.environ.get("GPT2_MODEL") else Path("gpt2")

STEP_RE = re.compile(
    r"private generation step:\s*"
    r"seq_len=(?P<seq_len>\d+),\s*"
    r"softmax=(?P<softmax>\w+),\s*"
    r"sqrt=(?P<sqrt>\w+),\s*"
    r"elapsed_s=(?P<elapsed>[\d.]+),\s*"
    r"bytes_per_party=(?P<bytes_mb>[\d.]+)\s*MB,\s*"
    r"rounds=(?P<rounds>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name-or-path", default=str(GPT2_MODEL))
    parser.add_argument("--len-data", type=int, default=64)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--softmax-config", default="ode_clip_i16")
    parser.add_argument("--sqrt-method", choices=["NR", "MLFormer"], default="NR")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = out_dir / "stdout.log"

    cmd = [
        "conda", "run", "-n", "shaft",
        "python", str(GPT2_SCRIPT),
        "--model_type=gpt2",
        f"--model_name_or_path={args.model_name_or_path}",
        f"--len_data={args.len_data}",
        f"--length={args.length}",
        f"--softmax_config={args.softmax_config}",
        f"--sqrt_method={args.sqrt_method}",
    ]
    if args.backend == "cpu":
        cmd.append("--use_cpu")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SHAFT_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "0" if args.backend == "gpu" else ""

    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    with stdout_log.open("w") as log:
        process = subprocess.Popen(
            cmd, cwd=str(QUEST_ROOT), env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        returncode = process.wait()

    captured = None
    for line in stdout_log.read_text().splitlines():
        match = STEP_RE.search(line)
        if match:
            captured = match.groupdict()
            break

    status = {
        "returncode": returncode,
        "backend": args.backend,
        "len_data": args.len_data,
        "length": args.length,
        "softmax_config": args.softmax_config,
        "sqrt_method": args.sqrt_method,
    }
    if captured:
        status["seq_len"] = int(captured["seq_len"])
        status["elapsed_s"] = float(captured["elapsed"])
        status["bytes_per_party_mb"] = float(captured["bytes_mb"])
        status["rounds"] = int(captured["rounds"])
        (out_dir / "summary.json").write_text(json.dumps(status, indent=2))
    else:
        print("WARNING: no 'private generation step' line found in stdout", file=sys.stderr)

    (out_dir / "replay_status.json").write_text(json.dumps(
        {"returncode": returncode, "log_path": str(stdout_log)}, indent=2
    ))
    print(json.dumps(status, indent=2))
    if returncode != 0:
        sys.exit(returncode)


if __name__ == "__main__":
    main()
