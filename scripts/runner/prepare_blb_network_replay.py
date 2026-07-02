#!/usr/bin/env python3
"""Prepare a no-run manifest for BLB-style throttled network replay on CPU and GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

from scripts.shared.blb_network_profiles import BLB_NETWORK_PROFILES  # noqa: E402

SHAFT_ROOT = get_shaft_root()
DATA_ROOT = get_data_root()
DEFAULT_ARTIFACT_ROOT = QUEST_ROOT / "artifacts" / "benchmark" / "blb_network_replay_20260617"
SUPERNET_ARTIFACT_ROOT = QUEST_ROOT / "artifacts" / "benchmark" / "supernet_bench_20260619"
THROTTLE_SCRIPT = QUEST_ROOT / "scripts" / "throttle_lo.sh"
LEN_MATRIX_RUNNER = QUEST_ROOT / "scripts" / "runner" / "run_len128_single_matrix.py"
GPT2_WRAPPER = QUEST_ROOT / "scripts" / "runner" / "wrap_gpt2_generation.py"
GPT2_SCRIPT = SHAFT_ROOT / "examples" / "text-generation" / "run_generation_private.py"
# GPT-2 weights: override via $GPT2_MODEL, otherwise let HF download to the cache.
GPT2_MODEL = Path(os.environ["GPT2_MODEL"]) if os.environ.get("GPT2_MODEL") else Path("gpt2")
BERT_LARGE_MODEL = DATA_ROOT / "bert-large-uncased"
SOFTMAX_OP_SCRIPT = QUEST_ROOT / "scripts" / "softmax" / "benchmark_len128_masked_softmax.py"
LAYERNORM_OP_SCRIPT = QUEST_ROOT / "scripts" / "shared" / "benchmark_layernorm_masked.py"


LEN_BENCHMARKS = {
    32: [
        ("shaft_original", "ode_clip_i16", "NR"),
        ("softmax_only", "scaled_k2_i8", "NR"),
        ("layernorm_only", "ode_clip_i16", "MLFormer"),
        ("both_optimized", "scaled_k2_i8", "MLFormer"),
    ],
    64: [
        ("shaft_original", "ode_clip_i16", "NR"),
        ("softmax_only", "scaled_k2_i8", "NR"),
        ("layernorm_only", "ode_clip_i16", "MLFormer"),
        ("both_optimized", "scaled_k2_i8", "MLFormer"),
        ("softmax_i12_only", "scaled_k2_i12", "NR"),
        ("i12_both_optimized", "scaled_k2_i12", "MLFormer"),
        ("softmax_i16_only", "scaled_k2_i16", "NR"),
        ("i16_both_optimized", "scaled_k2_i16", "MLFormer"),
    ],
    128: [
        ("shaft_original", "ode_clip_i16", "NR"),
        ("softmax_only", "scaled_k2_i8", "NR"),
        ("layernorm_only", "ode_clip_i16", "MLFormer"),
        ("both_optimized", "scaled_k2_i8", "MLFormer"),
        ("softmax_i12_only", "scaled_k2_i12", "NR"),
        ("i12_both_optimized", "scaled_k2_i12", "MLFormer"),
        ("softmax_i16_only", "scaled_k2_i16", "NR"),
        ("i16_both_optimized", "scaled_k2_i16", "MLFormer"),
    ],
    256: [
        ("shaft_original", "ode_clip_i16", "NR"),
        ("softmax_only", "scaled_k2_i8", "NR"),
        ("layernorm_only", "ode_clip_i16", "MLFormer"),
        ("both_optimized", "scaled_k2_i8", "MLFormer"),
        ("i16_both_optimized", "scaled_k2_i16", "MLFormer"),
    ],
}


GPT2_CASES = [
    {
        "name": "gpt2_len64_shaft",
        "len_data": 64,
        "softmax_config": "ode_clip_i16",
        "sqrt_method": "NR",
    },
    {
        "name": "gpt2_len64_both",
        "len_data": 64,
        "softmax_config": "scaled_k2_i8",
        "sqrt_method": "MLFormer",
    },
]


def build_len_matrix_cases(artifact_root: Path) -> list[dict]:
    cases = []
    for length, configs in LEN_BENCHMARKS.items():
        for case_name, softmax, sqrt in configs:
            for backend in ("gpu", "cpu"):
                for profile_name, profile in BLB_NETWORK_PROFILES.items():
                    out_dir = (
                        artifact_root
                        / backend
                        / profile_name
                        / f"len{length}"
                        / case_name
                    )
                    cmd = [
                        "conda",
                        "run",
                        "-n",
                        "shaft",
                        "python",
                        str(LEN_MATRIX_RUNNER),
                        "--backend",
                        backend,
                        "--softmax-config",
                        softmax,
                        "--sqrt-method",
                        sqrt,
                        "--max-length",
                        str(length),
                        "--max-samples",
                        "1",
                        "--output-dir",
                        str(out_dir),
                    ]
                    cases.append(
                        {
                            "suite": "bert_len_matrix",
                            "backend": backend,
                            "network_profile": profile_name,
                            "network_label": profile["label"],
                            "throttle_arg": profile["throttle_arg"],
                            "length": length,
                            "case_name": case_name,
                            "softmax_config": softmax,
                            "sqrt_method": sqrt,
                            "output_dir": str(out_dir),
                            "command": cmd,
                        }
                    )
    return cases


def build_gpt2_cases(artifact_root: Path) -> list[dict]:
    cases = []
    for spec in GPT2_CASES:
        for backend in ("gpu", "cpu"):
            for profile_name, profile in BLB_NETWORK_PROFILES.items():
                log_dir = artifact_root / backend / profile_name / spec["name"]
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    "shaft",
                    "python",
                    str(GPT2_SCRIPT),
                    "--model_type=gpt2",
                    f"--model_name_or_path={GPT2_MODEL}",
                    f"--len_data={spec['len_data']}",
                    "--length=1",
                    f"--softmax_config={spec['softmax_config']}",
                    f"--sqrt_method={spec['sqrt_method']}",
                ]
                if backend == "cpu":
                    cmd.append("--use_cpu")
                cases.append(
                    {
                        "suite": "gpt2_generation",
                        "backend": backend,
                        "network_profile": profile_name,
                        "network_label": profile["label"],
                        "throttle_arg": profile["throttle_arg"],
                        "length": spec["len_data"],
                        "case_name": spec["name"],
                        "softmax_config": spec["softmax_config"],
                        "sqrt_method": spec["sqrt_method"],
                        "output_dir": str(log_dir),
                        "command": cmd,
                        "env": {
                            "PYTHONPATH": str(SHAFT_ROOT),
                            "CUDA_VISIBLE_DEVICES": "0" if backend == "gpu" else "",
                        },
                    }
                )
    return cases


# --- Supernet benchmark campaign suites (20260619) -------------------------

BERT_LARGE_CASES = [
    ("shaft_original", "ode_clip_i16", "NR"),
    ("both_optimized", "scaled_k2_i8", "MLFormer"),
]


def build_bert_large_cases(artifact_root: Path) -> list[dict]:
    """BERT-large len128. shaft_original: GPU only; both_optimized: GPU + CPU."""
    cases = []
    for case_name, softmax, sqrt in BERT_LARGE_CASES:
        backends = ["gpu"] if case_name == "shaft_original" else ["gpu", "cpu"]
        for backend in backends:
            for profile_name, profile in BLB_NETWORK_PROFILES.items():
                out_dir = artifact_root / "bert_large" / backend / profile_name / case_name
                cmd = [
                    "conda", "run", "-n", "shaft", "python", str(LEN_MATRIX_RUNNER),
                    "--backend", backend,
                    "--softmax-config", softmax,
                    "--sqrt-method", sqrt,
                    "--max-length", "128",
                    "--max-samples", "1",
                    "--model-path", str(BERT_LARGE_MODEL),
                    "--output-dir", str(out_dir),
                ]
                cases.append({
                    "suite": "bert_large_len128",
                    "backend": backend,
                    "network_profile": profile_name,
                    "network_label": profile["label"],
                    "throttle_arg": profile["throttle_arg"],
                    "length": 128,
                    "model": "bert-large-uncased",
                    "case_name": case_name,
                    "softmax_config": softmax,
                    "sqrt_method": sqrt,
                    "output_dir": str(out_dir),
                    "command": cmd,
                })
    return cases


def build_microbench_cases(artifact_root: Path) -> list[dict]:
    """Single-op softmax + layernorm under throttle. Each script loops all cases."""
    cases = []
    ops = [
        ("softmax", SOFTMAX_OP_SCRIPT),
        ("layernorm", LAYERNORM_OP_SCRIPT),
    ]
    for op_name, script in ops:
        for backend in ("gpu", "cpu"):
            for profile_name, profile in BLB_NETWORK_PROFILES.items():
                out_dir = artifact_root / "microbench" / backend / profile_name / op_name
                out_dir.mkdir(parents=True, exist_ok=True)
                json_out = out_dir / f"{op_name}.json"
                cmd = [
                    "conda", "run", "-n", "shaft", "python", str(script),
                    "--max-length", "128",
                    "--layers", "0",
                    "--repeats", "1",
                    "--json-output", str(json_out),
                ]
                cases.append({
                    "suite": "microbench",
                    "op": op_name,
                    "backend": backend,
                    "network_profile": profile_name,
                    "network_label": profile["label"],
                    "throttle_arg": profile["throttle_arg"],
                    "length": 128,
                    "case_name": f"{op_name}_{backend}_{profile_name}",
                    "json_output": str(json_out),
                    "output_dir": str(out_dir),
                    "command": cmd,
                    "env": {
                        "PYTHONPATH": str(SHAFT_ROOT),
                        "CUDA_VISIBLE_DEVICES": "0" if backend == "gpu" else "",
                    },
                })
    return cases


def build_gpt2_wrapped_cases(artifact_root: Path) -> list[dict]:
    """GPT-2 len64 via wrap_gpt2_generation.py (captures summary.json)."""
    cases = []
    for spec in GPT2_CASES:
        for backend in ("gpu", "cpu"):
            for profile_name, profile in BLB_NETWORK_PROFILES.items():
                out_dir = artifact_root / "gpt2" / backend / profile_name / spec["name"]
                cmd = [
                    "conda", "run", "-n", "shaft", "python", str(GPT2_WRAPPER),
                    "--out-dir", str(out_dir),
                    "--model-name-or-path", str(GPT2_MODEL),
                    "--len-data", str(spec["len_data"]),
                    "--length", "1",
                    "--softmax-config", spec["softmax_config"],
                    "--sqrt-method", spec["sqrt_method"],
                    "--backend", backend,
                ]
                cases.append({
                    "suite": "gpt2_generation",
                    "backend": backend,
                    "network_profile": profile_name,
                    "network_label": profile["label"],
                    "throttle_arg": profile["throttle_arg"],
                    "length": spec["len_data"],
                    "case_name": spec["name"],
                    "softmax_config": spec["softmax_config"],
                    "sqrt_method": spec["sqrt_method"],
                    "output_dir": str(out_dir),
                    "command": cmd,
                })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-out",
        default=str(DEFAULT_ARTIFACT_ROOT / "manifest.json"),
        help="Where to write the manifest (and which artifact root to use).",
    )
    parser.add_argument(
        "--suite",
        choices=["legacy", "supernet", "all"],
        default="legacy",
        help="legacy=bert_len_matrix+raw-gpt2 (20260617); supernet=bert_large+microbench+gpt2-wrapped; all=both.",
    )
    args = parser.parse_args()

    out_path = Path(args.manifest_out)
    if args.suite in ("supernet", "all"):
        artifact_root = SUPERNET_ARTIFACT_ROOT
    else:
        artifact_root = DEFAULT_ARTIFACT_ROOT
    # If manifest-out is under a different dir, use that dir as the artifact root.
    if out_path.parent != DEFAULT_ARTIFACT_ROOT and out_path.parent != SUPERNET_ARTIFACT_ROOT:
        artifact_root = out_path.parent

    artifact_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    notes = []
    if args.suite in ("legacy", "all"):
        cases += build_len_matrix_cases(artifact_root)
        cases += build_gpt2_cases(artifact_root)
        notes += ["legacy suites: bert_len_matrix + gpt2_generation (raw script)."]
    if args.suite in ("supernet", "all"):
        cases += build_bert_large_cases(artifact_root)
        cases += build_microbench_cases(artifact_root)
        cases += build_gpt2_wrapped_cases(artifact_root)
        notes += [
            "supernet suites: bert_large_len128 + microbench + gpt2_generation (wrapped).",
            "BERT-large shaft_original is GPU-only; both_optimized is GPU+CPU per the campaign spec.",
            "Microbench runs the per-op scripts (which loop all cases internally) once per (op, backend, profile).",
        ]

    manifest = {
        "created_for": "BLB-style throttled network replay",
        "status": "prepared_only_not_executed",
        "artifact_root": str(artifact_root),
        "throttle_script": str(THROTTLE_SCRIPT),
        "network_profiles": BLB_NETWORK_PROFILES,
        "notes": notes + [
            "This manifest is preparation only. No runs have been executed yet.",
            "All runs are serialized because loopback tc/netem is global and the project forbids multi-GPU execution.",
            "GPU policy remains CUDA_VISIBLE_DEVICES=0 only.",
        ],
        "cases": cases,
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out_path)
    print(f"prepared_cases={len(cases)} suite={args.suite}")


if __name__ == "__main__":
    main()
