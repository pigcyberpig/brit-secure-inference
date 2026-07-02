"""Drive end-to-end BERT-base/BERT-large 2PC under tc/netem, softmax=scaled_k2_i10.

Runs shaft_original (ode_clip_i16/NR) and both_optimized (scaled_k2_i10/MLFormer)
on GPU under LAN (1G/0.3ms) and WAN (100M/80ms) throttle, recording measured
running_time_s and total_comm_bytes from the runner's summary JSON.

Usage: python scripts/runner/run_e2e_k2i10_throttled.py <sudo_pass>
Run `sudo -v` first or pass password as argv[1].
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

QUEST = QUEST_ROOT
THROTTLE = QUEST / "scripts" / "throttle_lo.sh"
RUNNER = QUEST / "scripts" / "runner" / "run_len128_single_matrix.py"
OUTDIR = QUEST / "artifacts" / "benchmark" / "e2e_k2i10_20260626"

_DATA = str(get_data_root())
SHAFT_PP = str(get_shaft_root())
BASE_MODEL = os.path.join(_DATA, "bert-base-cased-sst2")
LARGE_MODEL = os.path.join(_DATA, "bert-large-uncased")

PROFILES = [("lan", "blb-lan"), ("wan", "blb-wan3")]
# (case_name, softmax, sqrt)
CASES = [
    ("shaft_original", "ode_clip_i16", "NR"),
    ("both_optimized", "scaled_k2_i10", "MLFormer"),
]
MODELS = [("bert_base", BASE_MODEL), ("bert_large", LARGE_MODEL)]


def throttle(arg, sudo_pass):
    cmd = ["sudo", "-S", "bash", str(THROTTLE), arg]
    inp = (sudo_pass + "\n").encode() if sudo_pass else None
    subprocess.run(cmd, check=True, input=inp, capture_output=True)


def run_case(model_path, softmax, sqrt, out_dir):
    env = os.environ.copy()
    env["PYTHONPATH"] = SHAFT_PP
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = ["conda", "run", "-n", "shaft", "python", str(RUNNER),
           "--backend", "gpu", "--softmax-config", softmax, "--sqrt-method", sqrt,
           "--max-length", "128", "--max-samples", "1",
           "--model-path", model_path, "--output-dir", str(out_dir)]
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)


def read_summary(out_dir):
    sums = list(Path(out_dir).glob("summary_*.json"))
    if not sums:
        return None
    d = json.load(open(sums[0]))
    pf = d.get("private_forward", {})
    return {
        "running_time_s": d.get("running_time_s", 0),
        "total_comm_gb": round(pf.get("total_comm_bytes", 0) / 1e9, 4),
        "total_comm_rounds": pf.get("total_comm_rounds", 0),
    }


def main():
    sudo_pass = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}
    try:
        for net_name, arg in PROFILES:
            print(f"\n=== {net_name} ({arg}) ===", flush=True)
            throttle(arg, sudo_pass)
            results[net_name] = {}
            for model_name, model_path in MODELS:
                results[net_name][model_name] = {}
                for case_name, softmax, sqrt in CASES:
                    out_dir = OUTDIR / model_name / "gpu" / net_name / case_name
                    print(f"  {model_name}/{case_name}...", flush=True)
                    run_case(model_path, softmax, sqrt, out_dir)
                    r = read_summary(out_dir)
                    results[net_name][model_name][case_name] = r
                    if r:
                        print(f"    time={r['running_time_s']:.1f}s "
                              f"comm={r['total_comm_gb']}GB rounds={r['total_comm_rounds']}",
                              flush=True)
    finally:
        try:
            throttle("del", sudo_pass)
            print("\nthrottle cleared", flush=True)
        except Exception as e:
            print(f"WARN: clear failed: {e}", flush=True)

    json.dump(results, open(OUTDIR / "summary.json", "w"), indent=2)
    print(f"\n=== Summary → {OUTDIR/'summary.json'} ===")
    print(f"\n{'model':<12}{'case':<18}{'LAN(min)':>10}{'WAN(min)':>10}{'comm(GB)':>10}")
    for model_name, _ in MODELS:
        for case_name, _, _ in CASES:
            lan = results["lan"][model_name][case_name]
            wan = results["wan"][model_name][case_name]
            if lan and wan:
                print(f"{model_name:<12}{case_name:<18}"
                      f"{lan['running_time_s']/60:10.2f}"
                      f"{wan['running_time_s']/60:10.2f}"
                      f"{lan['total_comm_gb']:10.2f}")


if __name__ == "__main__":
    main()
