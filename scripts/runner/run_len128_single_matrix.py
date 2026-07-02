#!/usr/bin/env python3
"""Run one padded SST-2 private sample with chosen length, softmax, and LayerNorm."""

from __future__ import annotations

import argparse
import os
import sys

# Locate sibling scripts/ package and pull in path config (sets SHAFT_ROOT etc.).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
TEXT_CLASSIFICATION_ROOT = str(get_data_root())
DEFAULT_MODEL_PATH = os.path.join(TEXT_CLASSIFICATION_ROOT, "bert-base-cased-sst2")
DEFAULT_VALIDATION_FILE = os.path.join(TEXT_CLASSIFICATION_ROOT, "glue", "sst2", "validation.parquet")


def patched_main():
    from crypten.config import cfg
    from scripts.runner.run_glue_private_local import main as run_private_main

    method = os.environ.get("CRYPTEN_SQRT_METHOD", "NR")
    with cfg.temp_override({"functions.sqrt_method": method}):
        run_private_main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax-config", default="ode_clip_i16")
    parser.add_argument("--sqrt-method", choices=["NR", "MLFormer"], default="NR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task-name", default="sst2")
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CRYPTEN_SQRT_METHOD"] = args.sqrt_method

    conda_site_packages = os.path.join(
        os.path.dirname(sys.executable),
        "..",
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    conda_site_packages = os.path.abspath(conda_site_packages)

    for path in (conda_site_packages, SHAFT_ROOT, TEXT_CLASSIFICATION_ROOT, QUEST_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)

    sys.argv = [
        "scripts/runner/run_glue_private_local.py",
        "--task_name",
        args.task_name,
        "--model_name_or_path",
        args.model_path,
        "--validation_file",
        args.validation_file,
        "--max_length",
        str(args.max_length),
        "--len_data",
        str(args.max_length),
        "--pad_to_max_length",
        "--per_device_eval_batch_size",
        "1",
        "--softmax_config",
        args.softmax_config,
        "--sqrt_method",
        args.sqrt_method,
        "--max_samples",
        str(args.max_samples),
        "--output_dir",
        args.output_dir,
    ]
    if args.backend == "gpu":
        sys.argv.extend(["--cuda_device", "0"])

    from multiprocess_launcher import MultiProcessLauncher

    launcher = MultiProcessLauncher(2, patched_main)
    launcher.start()
    launcher.join()
    launcher.terminate()


if __name__ == "__main__":
    main()
