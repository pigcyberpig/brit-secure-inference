#!/usr/bin/env python3
"""Execute the prepared BLB-style throttled network replay manifest serially."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT  # noqa: E402

MANIFEST_PATH = QUEST_ROOT / "artifacts" / "benchmark" / "blb_network_replay_20260617" / "manifest.json"
THROTTLE_SCRIPT = QUEST_ROOT / "scripts" / "throttle_lo.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--backend", choices=["gpu", "cpu", "all"], default="all")
    parser.add_argument("--suite", choices=["all", "bert_len_matrix", "gpt2_generation"], default="all")
    parser.add_argument("--network-profile", default="all")
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    return payload, payload["cases"]


def filter_cases(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = []
    case_names = None
    if args.case_name:
        case_names = set(args.case_name)
    for case in cases:
        if args.backend != "all" and case["backend"] != args.backend:
            continue
        if args.suite != "all" and case["suite"] != args.suite:
            continue
        if args.network_profile != "all" and case["network_profile"] != args.network_profile:
            continue
        if case_names is not None and case["case_name"] not in case_names:
            continue
        if args.length is not None and int(case.get("length", -1)) != args.length:
            continue
        if not args.rerun_completed:
            status_path = Path(case["output_dir"]) / "replay_status.json"
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text())
                    if int(status.get("returncode", 1)) == 0:
                        continue
                except Exception:
                    pass
        selected.append(case)
    if args.case_limit is not None:
        selected = selected[: args.case_limit]
    return selected


def run_logged(command: list[str], *, env: dict[str, str] | None, cwd: Path, log_path: Path) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.wait()


def apply_throttle(profile_arg: str, qdisc_path: Path) -> None:
    subprocess.run(["bash", str(THROTTLE_SCRIPT), profile_arg], check=True, cwd=str(QUEST_ROOT))
    result = subprocess.run(
        ["bash", str(THROTTLE_SCRIPT), "show"],
        check=True,
        cwd=str(QUEST_ROOT),
        capture_output=True,
        text=True,
    )
    qdisc_path.write_text(result.stdout)


def clear_throttle() -> None:
    subprocess.run(["bash", str(THROTTLE_SCRIPT), "del"], check=False, cwd=str(QUEST_ROOT))


def main() -> None:
    args = parse_args()
    manifest_payload, cases = load_cases(Path(args.manifest))
    selected = filter_cases(cases, args)

    print(f"selected_cases={len(selected)}")
    if args.dry_run:
        for case in selected:
            print(json.dumps(case, ensure_ascii=False))
        return

    results = []
    try:
        for idx, case in enumerate(selected, start=1):
            out_dir = Path(case["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            run_log = out_dir / "replay_run.log"
            qdisc_log = out_dir / "qdisc.txt"
            status_path = out_dir / "replay_status.json"

            print(
                f"[{idx}/{len(selected)}] backend={case['backend']} "
                f"profile={case['network_profile']} suite={case['suite']} case={case['case_name']}"
            )
            apply_throttle(case["throttle_arg"], qdisc_log)
            start = time.time()
            returncode = run_logged(
                case["command"],
                env=case.get("env"),
                cwd=QUEST_ROOT,
                log_path=run_log,
            )
            elapsed = time.time() - start
            status = {
                "case": case,
                "returncode": returncode,
                "wall_time_s": elapsed,
                "qdisc_path": str(qdisc_log),
                "log_path": str(run_log),
            }
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
            results.append(status)
            if returncode != 0:
                print(f"case failed with returncode={returncode}: {case['case_name']}")
                break
    finally:
        clear_throttle()

    summary_path = Path(args.manifest).with_name("replay_execution_status.json")
    summary_path.write_text(
        json.dumps(
            {
                "manifest_notes": manifest_payload.get("notes", []),
                "executed_cases": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(summary_path)


if __name__ == "__main__":
    main()
