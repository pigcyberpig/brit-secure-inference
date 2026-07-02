"""Single-operator LayerNorm 2PC benchmark: NR (SHAFT) vs MLFormer (our method).

Mirrors scripts/softmax/benchmark_len128_masked_softmax.py: load a real padded
SST-2 sample, take the residual-stream hidden state feeding each target BERT
layer as the LayerNorm input, and measure ONE crypten.nn.LayerNormalization
call per (layer x case) under each sqrt_method. Records comm bytes / rounds /
compute time plus the three network-profile time estimates (LAN / WAN-4ms /
WAN-80ms).

Comm/time are shape-determined, so gamma=1/beta=0 (a valid LayerNorm config)
gives the same comm/time as any real gamma/beta; secure multiplication cost is
fixed per element regardless of value. The hidden-state *input* is kept real to
match the softmax benchmark's methodology.
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import crypten
import crypten.communicator as comm
import crypten.nn as nn


def resolve_device():
    """Mirror run_glue_private_local: one GPU per CrypTen rank on cuda:0."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return "cuda:0"
    return "cpu"


LAN_BPS = 3_000_000_000
LAN_LATENCY_S = 0.0005
WAN_MID_BPS = 400_000_000
WAN_MID_LATENCY_S = 0.004
WAN_HARD_BPS = 100_000_000
WAN_HARD_LATENCY_S = 0.080


def load_padded_sample(dataset_path, model_path, max_length, offset):
    dataset = load_dataset(
        "parquet",
        data_files={"validation": f"{dataset_path}/validation.parquet"},
        split="validation",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer(
        dataset[offset]["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def get_layernorm_inputs(model_path, dataset_path, max_length, offset, layers):
    """Return residual-stream hidden state (input to each target layer) + valid_len."""
    encoded = load_padded_sample(dataset_path, model_path, max_length, offset)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    bert = model.bert
    hidden_size = bert.config.hidden_size
    with torch.no_grad():
        hidden = bert.embeddings(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded.get("token_type_ids"),
        )
        attention_mask = bert.get_extended_attention_mask(
            encoded["attention_mask"], encoded["input_ids"].shape
        )
        outputs = []
        for layer_idx, layer in enumerate(bert.encoder.layer):
            if layer_idx in layers:
                outputs.append(
                    {
                        "layer": layer_idx,
                        "input": hidden.reshape(max_length, hidden_size).contiguous(),
                        "valid_len": int(encoded["attention_mask"].sum().item()),
                    }
                )
            hidden = layer(hidden, attention_mask=attention_mask)[0]
    return outputs, hidden_size


def layernorm_forward(input_enc, scale_enc, bias_enc):
    """Replicates crypten.nn.LayerNormalization.forward([input, scale, bias])."""
    mean = input_enc.mean(dim=-1, keepdim=True)
    inv_sd = input_enc.var(dim=-1, keepdim=True).inv_sqrt()
    return (input_enc - mean) * inv_sd * scale_enc + bias_enc


def measure(input_enc, scale_enc, bias_enc, reference, repeats, valid_len):
    records = []
    last = None
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        result = layernorm_forward(input_enc, scale_enc, bias_enc)
        wall = time.perf_counter() - start
        stats = crypten.get_communication_stats()
        last = result.get_plain_text().cpu()
        records.append(
            {
                "wall": wall,
                "comm_time": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )
    diff = (last - reference).abs()
    # Split valid vs padding rows: rows are tokens, first valid_len are valid.
    valid_diff = diff[:valid_len]
    padded_diff = diff[valid_len:]
    avg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    avg["compute"] = max(avg["wall"] - avg["comm_time"], 0.0)
    avg["max_abs_diff"] = diff.max().item()
    avg["mean_abs_diff"] = diff.mean().item()
    avg["valid_max_abs_diff"] = valid_diff.max().item()
    avg["valid_mean_abs_diff"] = valid_diff.mean().item()
    avg["padded_max_abs_diff"] = padded_diff.max().item() if padded_diff.numel() else 0.0
    avg["padded_mean_abs_diff"] = padded_diff.mean().item() if padded_diff.numel() else 0.0
    return avg


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(model_path, dataset_path, max_length, offset, layer_ids, repeats):
    device = resolve_device()
    # NOTE: BOLT (S&P'24) and BumbleBee (NDSS'25) both compute LayerNorm's
    # reciprocal square root via Newton iteration (BumbleBee reuses [44], BOLT
    # reuses SIRNN), which is exactly the NR case below. They are not listed
    # separately to avoid duplicate byte-identical rows.
    cases = [
        ("NR", {"functions.sqrt_method": "NR"}),
        ("MLFormer", {"functions.sqrt_method": "MLFormer"}),
    ]
    items, hidden_size = get_layernorm_inputs(
        model_path, dataset_path, max_length, offset, set(layer_ids)
    )
    # gamma=1 / beta=0: valid LayerNorm config; comm/time are value-independent.
    scale = torch.ones(hidden_size)
    bias = torch.zeros(hidden_size)
    outputs = []
    for item in items:
        inp = item["input"]
        reference = torch.nn.functional.layer_norm(
            inp, (hidden_size,), scale, bias, eps=1e-12
        )
        input_enc = crypten.cryptensor(inp, src=0).to(device)
        scale_enc = crypten.cryptensor(scale, src=0).to(device)
        bias_enc = crypten.cryptensor(bias, src=0).to(device)
        results = []
        for name, cfg_override in cases:
            with crypten.cfg.temp_override(cfg_override):
                results.append(
                    (name, measure(input_enc, scale_enc, bias_enc, reference, repeats, item["valid_len"]))
                )
        outputs.append(
            {
                "layer": item["layer"],
                "shape": tuple(inp.shape),
                "valid_len": item["valid_len"],
                "results": results,
            }
        )
    return {"rank": comm.get().get_rank(), "outputs": outputs}


def estimate_time(compute_s, comm_bytes, rounds, bandwidth_bps, latency_s):
    return compute_s + 2 * comm_bytes / (bandwidth_bps / 8) + rounds * latency_s


def summarize(rank_outputs):
    summary = []
    for layer_pos, first in enumerate(rank_outputs[0]["outputs"]):
        print(
            f"\nlayer={first['layer']} shape={first['shape']} "
            f"valid_len={first['valid_len']}"
        )
        print(
            f"{'case':<10} {'comp_ms':>9} {'comm_MB':>9} {'rounds':>7} "
            f"{'LAN_ms':>9} {'WAN4_ms':>9} {'WAN80_ms':>10} "
            f"{'max_diff':>11} {'mean_diff':>11}"
        )
        by_case = {}
        for rank_output in rank_outputs:
            for name, result in rank_output["outputs"][layer_pos]["results"]:
                by_case.setdefault(name, []).append(result)
        for name, items in by_case.items():
            compute = max(item["compute"] for item in items)
            comm_bytes = sum(item["bytes"] for item in items) / 2
            rounds = max(item["rounds"] for item in items)
            lan = estimate_time(compute, comm_bytes, rounds, LAN_BPS, LAN_LATENCY_S)
            wan_mid = estimate_time(compute, comm_bytes, rounds, WAN_MID_BPS, WAN_MID_LATENCY_S)
            wan_hard = estimate_time(
                compute, comm_bytes, rounds, WAN_HARD_BPS, WAN_HARD_LATENCY_S
            )
            summary.append(
                {
                    "layer": first["layer"],
                    "shape": first["shape"],
                    "valid_len": first["valid_len"],
                    "case": name,
                    "compute_time_s": compute,
                    "comm_bytes": comm_bytes,
                    "comm_mb": comm_bytes / 1_000_000,
                    "rounds": rounds,
                    "comm_time_s": max(item["comm_time"] for item in items),
                    "wall_time_s": max(item["wall"] for item in items),
                    "lan_3g_0p5ms_time_s": lan,
                    "wan_400m_4ms_time_s": wan_mid,
                    "wan_100m_80ms_time_s": wan_hard,
                    "max_abs_diff": max(item["max_abs_diff"] for item in items),
                    "mean_abs_diff": max(item["mean_abs_diff"] for item in items),
                    "valid_max_abs_diff": max(item["valid_max_abs_diff"] for item in items),
                    "valid_mean_abs_diff": max(item["valid_mean_abs_diff"] for item in items),
                    "padded_max_abs_diff": max(item["padded_max_abs_diff"] for item in items),
                    "padded_mean_abs_diff": max(item["padded_mean_abs_diff"] for item in items),
                }
            )
            print(
                f"{name:<10} {compute * 1000:9.3f} {comm_bytes / 1_000_000:9.4f} "
                f"{rounds:7.1f} {lan * 1000:9.3f} {wan_mid * 1000:9.3f} "
                f"{wan_hard * 1000:10.3f} "
                f"{max(item['max_abs_diff'] for item in items):11.5f} "
                f"{max(item['mean_abs_diff'] for item in items):11.5f}"
            )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.environ.get("DATA_ROOT", ""), "bert-base-cased-sst2"),
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.environ.get("DATA_ROOT", ""), "glue", "sst2"),
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    outputs = run_rank(
        args.model_path,
        args.dataset_path,
        args.max_length,
        args.offset,
        args.layers,
        args.repeats,
    )
    summary = summarize(outputs)
    if args.json_output:
        with open(args.json_output, "w") as handle:
            json.dump({"results": summary}, handle, indent=2)


if __name__ == "__main__":
    main()
