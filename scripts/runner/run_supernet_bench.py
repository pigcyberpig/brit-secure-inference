#!/usr/bin/env python3
"""Profile-grouped, sudo-free serial driver for the supernet benchmark campaign.

Unlike run_blb_network_replay.py, this driver does NOT apply the tc/netem
throttle itself (that needs sudo). Instead, for each network profile it:
  1. prints the exact `! sudo bash scripts/throttle_lo.sh <arg>` the user must run,
  2. runs `bash scripts/throttle_lo.sh show` (read-only, no sudo) so the active
     qdisc can be eyeballed,
  3. optionally asserts the active profile matches --confirm-throttle,
  4. runs every filtered case for that profile serially, each logged to
     replay_run.log with a replay_status.json (skipped if already returncode==0).

Throttle is global and GPU0 is exclusive, so all runs are strictly serial.
"""

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

DEFAULT_MANIFEST = QUEST_ROOT / "artifacts" / "benchmark" / "supernet_bench_20260619" / "manifest.json"
THROTTLE_SCRIPT = QUEST_ROOT / "scripts" / "throttle_lo.sh"
PROFILES = [
    "lan_3g_0p3ms",
    "blb_lan",
    "blb_wan1",
    "blb_wan2",
    "blb_wan3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--backend", choices=["gpu", "cpu", "all"], default="all")
    parser.add_argument("--suite", choices=["all", "microbench", "bert_large_len128", "gpt2_generation"], default="all")
    parser.add_argument("--network-profile", default="all")
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--confirm-throttle", default=None,
                        help="Throttle arg expected to be active (e.g. blb-wan1). Aborts on mismatch.")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Do not prompt; just run.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["cases"]


def filter_cases(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = []
    case_names = set(args.case_name) if args.case_name else None
    for case in cases:
        if args.backend != "all" and case["backend"] != args.backend:
            continue
        if args.suite != "all" and case["suite"] != args.suite:
            continue
        if args.network_profile != "all" and case["network_profile"] != args.network_profile:
            continue
        if case_names is not None and case["case_name"] not in case_names:
            continue
        selected.append(case)
    return selected


def show_qdisc() -> str:
    result = subprocess.run(
        ["bash", str(THROTTLE_SCRIPT), "show"],
        cwd=str(QUEST_ROOT), capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def qdisc_matches(qdisc_text: str, throttle_arg: str) -> bool:
    """Best-effort check that the active qdisc matches the expected profile.

    tc prints normalized units (3Gbit, 249us) that differ from the throttle_lo.sh
    input tokens (3000mbit, 0.15msec), so we parse the rate (Mbps) and one-way
    delay (us) out of the qdisc text and compare against the profile spec.
    """
    import re
    spec = {
        "lan3g03": (3000, 150), "blb-lan": (1000, 150), "blb-wan1": (400, 2000),
        "blb-wan2": (100, 2000), "blb-wan3": (100, 40000),
    }
    if throttle_arg not in spec:
        return True  # unknown arg, skip check
    exp_rate_mbps, exp_delay_us = spec[throttle_arg]

    def parse_rate(text: str):
        m = re.search(r"rate\s+(\d+(?:\.\d+)?)\s*([KMG]?)bit", text)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2)
        return val * {"": 1e-6, "K": 1e-3, "M": 1.0, "G": 1e3}[unit]

    def parse_delay(text: str):
        m = re.search(r"delay\s+(\d+(?:\.\d+)?)\s*([munp]?s|sec|msec|usec)", text)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2)
        to_us = {"s": 1e6, "sec": 1e6, "msec": 1e3, "ms": 1e3, "us": 1.0, "usec": 1.0, "ns": 1e-3}
        return val * to_us.get(unit, 1.0)

    rate = parse_rate(qdisc_text)
    delay = parse_delay(qdisc_text)
    if rate is None or delay is None:
        return False
    return abs(rate - exp_rate_mbps) / max(exp_rate_mbps, 1) < 0.05 and \
        abs(delay - exp_delay_us) / max(exp_delay_us, 1) < 0.1


def run_logged(command: list[str], env: dict | None, cwd: Path, log_path: Path) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(shlex.quote(p) for p in command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command, cwd=str(cwd), env=merged,
            stdout=log_file, stderr=subprocess.STDOUT, text=True,
        )
        return process.wait()


def already_done(case: dict) -> bool:
    status_path = Path(case["output_dir"]) / "replay_status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text())
        return int(status.get("returncode", 1)) == 0
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.manifest))
    selected = filter_cases(cases, args)

    print(f"selected_cases={len(selected)}")
    if args.dry_run:
        for case in selected:
            print(json.dumps({
                "suite": case["suite"], "backend": case["backend"],
                "profile": case["network_profile"], "case": case["case_name"],
                "throttle_arg": case["throttle_arg"],
            }, ensure_ascii=False))
        return
    if not selected:
        print("nothing to run")
        return

    # Group by profile, preserving the canonical cheap->expensive order.
    by_profile: dict[str, list[dict]] = {p: [] for p in PROFILES}
    for case in selected:
        by_profile.setdefault(case["network_profile"], []).append(case)

    for profile in [p for p in PROFILES if by_profile.get(p)]:
        group = by_profile[profile]
        throttle_arg = group[0]["throttle_arg"]
        print("\n" + "=" * 72)
        print(f"PROFILE: {profile}  ({len(group)} cases)")
        print(f"  Throttle command (USER, needs sudo):")
        print(f"    ! sudo bash scripts/throttle_lo.sh {throttle_arg}")
        print("  Current qdisc:")
        qdisc = show_qdisc()
        for line in qdisc.splitlines():
            print(f"    {line}")

        if args.confirm_throttle:
            if args.confirm_throttle != throttle_arg:
                print(f"  ABORT: --confirm-throttle={args.confirm_throttle} != active {throttle_arg}")
                sys.exit(2)
            matched = qdisc_matches(qdisc, throttle_arg)
            if not matched:
                print(f"  NOTE: qdisc does not strictly match {throttle_arg} (tc normalizes units).")
                print(f"        Proceeding -- visually confirm the qdisc above is correct.")
        elif not args.yes:
            resp = input(f"  Throttle {throttle_arg} applied and qdisc correct? [y/N] ")
            if resp.strip().lower() not in ("y", "yes"):
                print("  skipping this profile group")
                continue

        for idx, case in enumerate(group, start=1):
            out_dir = Path(case["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            run_log = out_dir / "replay_run.log"
            qdisc_path = out_dir / "qdisc.txt"
            status_path = out_dir / "replay_status.json"
            tag = f"{case['suite']}/{case['backend']}/{case['case_name']}"

            if not args.rerun_completed and already_done(case):
                print(f"  [{idx}/{len(group)}] SKIP (done): {tag}")
                continue

            qdisc_path.write_text(qdisc)
            print(f"  [{idx}/{len(group)}] RUN: {tag}")
            start = time.time()
            returncode = run_logged(case["command"], case.get("env"), QUEST_ROOT, run_log)
            elapsed = time.time() - start
            status = {
                "case": {k: case[k] for k in ("suite", "backend", "network_profile", "case_name")},
                "returncode": returncode,
                "wall_time_s": elapsed,
                "qdisc_path": str(qdisc_path),
                "log_path": str(run_log),
            }
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
            if returncode != 0:
                print(f"  -> FAILED returncode={returncode} (see {run_log})")
                print("  Stopping this profile group. Fix and rerun (skip-logic resumes).")
                break

    print("\nDone. Clear throttle when finished: ! sudo bash scripts/throttle_lo.sh del")


if __name__ == "__main__":
    main()
