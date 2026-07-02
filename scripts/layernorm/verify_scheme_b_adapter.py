#!/usr/bin/env python3
"""Minimal verification for the reusable public-mask-aware BERT adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
SHARED_DIR = str(QUEST_ROOT / "scripts" / "shared")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

import crypten as ct
import torch
from crypten.config import cfg

from public_masked_attention import masked_softmax


def softmax_override(name: str) -> dict:
    if name == "ode_clip_i16":
        return {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": True,
            "functions.softmax_ode_iter_num": 16,
        }
    if name.startswith("scaled_k") and "_i" in name:
        return {"functions.softmax_method": name}
    raise ValueError(f"unknown softmax config: {name}")


def main() -> None:
    output_dir = Path("artifacts/experiment/public_mask_batching_20260605")
    output_dir.mkdir(parents=True, exist_ok=True)

    ct.init()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0, -10000.0, -10000.0]],
            [[1.5, 0.5, -10000.0, -10000.0, -10000.0]],
        ],
        device=device,
    )
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)

    with cfg.temp_override({"functions.sqrt_method": "MLFormer"}):
        with cfg.temp_override(softmax_override("scaled_k2_i8")):
            additive = ct.cryptensor(logits).softmax(-1).get_plain_text()
            masked = masked_softmax(
                ct.cryptensor(logits),
                mask,
                -1,
                mask_layout="attention_key",
            ).get_plain_text()

    broadcast_mask = mask[:, None, :].to(device)
    payload = {
        "softmax_config": "scaled_k2_i8",
        "sqrt_method": "MLFormer",
        "additive_pad_mass_max": float((additive * (1.0 - broadcast_mask)).sum(-1).max().cpu()),
        "masked_pad_mass_max": float((masked * (1.0 - broadcast_mask)).sum(-1).max().cpu()),
        "masked_row_sum_max_abs_err": float((masked.sum(-1) - 1.0).abs().max().cpu()),
    }
    output = output_dir / "scheme_b_adapter_check.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
