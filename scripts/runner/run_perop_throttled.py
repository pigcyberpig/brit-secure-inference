"""Drive per-operator softmax/layernorm benchmarks under three tc/netem network
profiles, recording the ACTUAL measured wall time (not the formula estimate).

For each profile, we apply tc throttle on loopback, run the existing single-op
benchmark scripts, and read the measured `wall_time_s` field from their JSON
output. The wall time under throttle reflects the real network latency.

Network profiles (matches paper §6.2.3):
  LAN   = blb-lan   (1 Gbps, 0.3 ms RTT)
  WAN-1 = blb-wan1  (400 Mbps, 4 ms RTT)
  WAN-2 = blb-wan3  (100 Mbps, 80 ms RTT)

Usage:
  sudo -v  # cache sudo creds first
  python scripts/runner/run_perop_throttled.py
The script invokes sudo via the throttle script; run `sudo -v` in the shell
first so the password prompt is already satisfied, or pass sudo via stdin.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root  # noqa: E402

QUEST = QUEST_ROOT
THROTTLE = QUEST / "scripts" / "throttle_lo.sh"
SHAFT_PP = str(get_shaft_root())
SOFTMAX_SCRIPT = QUEST / "scripts/softmax/benchmark_len128_masked_softmax.py"
LN_SCRIPT = QUEST / "scripts/shared/benchmark_layernorm_masked.py"
OUTDIR = QUEST / "artifacts/benchmark/perop_throttled_20260626"

PROFILES = [
    ("LAN", "blb-lan"),
    ("WAN1", "blb-wan1"),
    ("WAN2", "blb-wan3"),
]


def run(cmd, env=None):
    """Run a command, return CompletedProcess (raise on failure)."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def throttle(profile_arg, sudo_pass=None):
    cmd = ["sudo", "-S", "bash", str(THROTTLE), profile_arg]
    inp = (sudo_pass + "\n").encode() if sudo_pass else None
    subprocess.run(cmd, check=True, input=inp, capture_output=True)


def run_bench(script, extra_args, out_json, sudo_pass=None):
    """Run one benchmark script under the currently-applied throttle."""
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = SHAFT_PP
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = ["conda", "run", "-n", "shaft", "python", str(script)] + extra_args + \
          ["--json-output", str(out_json)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return res


def collect_walls(json_path):
    """Return {case: (layer, comm_mb, rounds, wall_time_s)} per layer-0 case."""
    d = json.load(open(json_path))
    out = {}
    for r in d["results"]:
        if r["layer"] != 0:
            continue
        out[r["case"]] = {
            "comm_mb": r["comm_mb"],
            "rounds": int(r["rounds"]),
            "wall_time_s": r["wall_time_s"],
            "compute_time_s": r["compute_time_s"],
        }
    return out


def main():
    sudo_pass = sys.argv[1] if len(sys.argv) > 1 else None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # softmax scales: (label, synthetic_heads). BERT-base=12, BERT-large=16.
    SM_SCALES = [("bertbase_h12", "12"), ("bertlarge_h16", "16")]
    all_results = {}
    try:
        for name, arg in PROFILES:
            print(f"\n=== {name} ({arg}) ===", flush=True)
            throttle(arg, sudo_pass)
            all_results[name] = {"softmax": {}, "layernorm": {}}
            for label, heads in SM_SCALES:
                sm_json = OUTDIR / f"softmax_{label}_{name}.json"
                print(f"  softmax {label}...", flush=True)
                run_bench(SOFTMAX_SCRIPT,
                          ["--synthetic-heads", heads, "--max-length", "128",
                           "--layers", "0", "--repeats", "3"], sm_json)
                all_results[name]["softmax"][label] = collect_walls(sm_json)
            ln_json = OUTDIR / f"layernorm_{name}.json"
            print(f"  layernorm...", flush=True)
            run_bench(LN_SCRIPT, ["--max-length", "128", "--layers", "0", "--repeats", "3"], ln_json)
            all_results[name]["layernorm"] = collect_walls(ln_json)
            print(f"  {name} done", flush=True)
    finally:
        try:
            throttle("del", sudo_pass)
            print("\nthrottle cleared", flush=True)
        except Exception as e:
            print(f"\nWARN: failed to clear throttle: {e}", flush=True)

    summary_path = OUTDIR / "measured_walls_multiscale.json"
    json.dump(all_results, open(summary_path, "w"), indent=2)
    print(f"\n=== Summary written to {summary_path} ===")
    for label in ["bertbase_h12", "bertlarge_h16"]:
        print(f"\nSoftmax {label} (layer 0, measured wall time s):")
        print(f"{'case':<16}{'comm_MB':>9}{'rounds':>7}{'LAN':>9}{'WAN1':>9}{'WAN2':>9}")
        cases = list(all_results["LAN"]["softmax"][label].keys())
        for c in cases:
            try:
                comm = all_results["LAN"]["softmax"][label][c]["comm_mb"]
                rnd = all_results["LAN"]["softmax"][label][c]["rounds"]
                lan = all_results["LAN"]["softmax"][label][c]["wall_time_s"]
                w1 = all_results["WAN1"]["softmax"][label][c]["wall_time_s"]
                w2 = all_results["WAN2"]["softmax"][label][c]["wall_time_s"]
                print(f"{c:<16}{comm:9.2f}{rnd:7d}{lan:9.3f}{w1:9.3f}{w2:9.3f}")
            except KeyError:
                pass
    print("\nLayerNorm 128x768 (layer 0, measured wall time s):")
    print(f"{'case':<16}{'comm_MB':>9}{'rounds':>7}{'LAN':>9}{'WAN1':>9}{'WAN2':>9}")
    for c in all_results["LAN"]["layernorm"]:
        try:
            comm = all_results["LAN"]["layernorm"][c]["comm_mb"]
            rnd = all_results["LAN"]["layernorm"][c]["rounds"]
            lan = all_results["LAN"]["layernorm"][c]["wall_time_s"]
            w1 = all_results["WAN1"]["layernorm"][c]["wall_time_s"]
            w2 = all_results["WAN2"]["layernorm"][c]["wall_time_s"]
            print(f"{c:<16}{comm:9.2f}{rnd:7d}{lan:9.4f}{w1:9.4f}{w2:9.4f}")
        except KeyError:
            pass


if __name__ == "__main__":
    main()
