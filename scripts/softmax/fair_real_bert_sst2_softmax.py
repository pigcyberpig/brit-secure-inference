import argparse
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import crypten
import crypten.communicator as comm


LAN_BPS = 1_000_000_000
LAN_LATENCY_S = 0.0005
WAN_BPS = 400_000_000
WAN_LATENCY_S = 0.004


def load_sst2_batch(dataset_path, model_path, max_length, batch_size, offset):
    data_files = {"validation": f"{dataset_path}/validation.parquet"}
    dataset = load_dataset("parquet", data_files=data_files, split="validation")
    sentences = [dataset[i]["sentence"] for i in range(offset, offset + batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def split_heads(x, num_heads):
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)


def get_bert_logits(
    model_path, dataset_path, max_length, batch_size, offset, layers, target_p99
):
    encoded = load_sst2_batch(dataset_path, model_path, max_length, batch_size, offset)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    bert = model.bert
    with torch.no_grad():
        hidden = bert.embeddings(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded.get("token_type_ids"),
        )

        outputs = []
        for layer_idx, layer in enumerate(bert.encoder.layer):
            attn = layer.attention.self
            query = split_heads(attn.query(hidden), attn.num_attention_heads)
            key = split_heads(attn.key(hidden), attn.num_attention_heads)
            logits = torch.matmul(query, key.transpose(-1, -2))
            logits = logits / math.sqrt(attn.attention_head_size)
            rows = []
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for sample_idx, valid_len in enumerate(lengths):
                valid_len = int(valid_len)
                sample_logits = logits[sample_idx, :, :valid_len, :valid_len]
                rows.append(sample_logits.reshape(-1, valid_len))
            min_len = min(row.size(1) for row in rows)
            # Keep a rectangular tensor for one batched MPC call.
            logits = torch.cat([row[:, :min_len] for row in rows], dim=0)
            if target_p99 is not None:
                finite = logits > -1000
                centered = logits.clone()
                row_mean = torch.where(finite, logits, torch.zeros_like(logits)).sum(
                    dim=-1, keepdim=True
                ) / finite.sum(dim=-1, keepdim=True).clamp_min(1)
                centered = torch.where(finite, logits - row_mean, logits)
                p99 = torch.quantile(centered[finite].abs(), 0.99).clamp_min(1e-6)
                logits = torch.where(finite, centered * (target_p99 / p99), logits)
            if layer_idx in layers:
                finite_logits = logits[logits > -1000]
                outputs.append(
                    {
                        "layer": layer_idx,
                        "logits": logits.contiguous(),
                        "min": finite_logits.min().item(),
                        "max": finite_logits.max().item(),
                        "std": finite_logits.std().item(),
                    }
                )
            extended_attention_mask = bert.get_extended_attention_mask(
                encoded["attention_mask"], encoded["input_ids"].shape
            )
            hidden = layer(hidden, attention_mask=extended_attention_mask)[0]
    return outputs


def pow_int(tensor, exponent):
    result = None
    base = tensor
    current = exponent
    while current:
        if current & 1:
            result = base if result is None else result * base
        current >>= 1
        if current:
            base = base * base
    return result


def scaled_k_softmax(tensor, dim, scale, iters, clip=False):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / scale
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": clip,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        probs = scaled.softmax(dim=dim)
    powered = pow_int(probs, scale)
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return powered * powered.sum(dim=dim, keepdim=True).reciprocal()


def ode_softmax(tensor, dim, clip):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": clip,
            "functions.softmax_ode_iter_num": 16,
        }
    ):
        return tensor.softmax(dim=dim)


def build_cases(scales, scaled_iters, scaled_clip_iters):
    cases = [("ode_clip_i16", lambda x: ode_softmax(x, -1, True))]
    for scale in scales:
        for iters in scaled_iters:
            cases.append(
                (
                    f"scaled_k{scale}_i{iters}",
                    lambda x, scale=scale, iters=iters: scaled_k_softmax(
                        x, -1, scale, iters
                    ),
                )
            )
        for iters in scaled_clip_iters:
            cases.append(
                (
                    f"scaled_clip_k{scale}_i{iters}",
                    lambda x, scale=scale, iters=iters: scaled_k_softmax(
                        x, -1, scale, iters, clip=True
                    ),
                )
            )
    return cases


def measure(encrypted, reference, fn, repeats):
    records = []
    last = None
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        result = fn(encrypted)
        wall = time.perf_counter() - start
        stats = crypten.get_communication_stats()
        last = result.get_plain_text()
        records.append(
            {
                "wall": wall,
                "comm_time": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )
    diff = (last - reference).abs()
    avg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    avg["compute"] = max(avg["wall"] - avg["comm_time"], 0.0)
    avg["max_abs"] = diff.max().item()
    avg["mean_abs"] = diff.mean().item()
    avg["row_sum"] = (last.sum(dim=-1) - 1).abs().max().item()
    return avg


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(
    model_path,
    dataset_path,
    max_length,
    batch_size,
    offset,
    layer_ids,
    repeats,
    target_p99,
    scales,
    scaled_iters,
    scaled_clip_iters,
):
    layer_logits = get_bert_logits(
        model_path,
        dataset_path,
        max_length,
        batch_size,
        offset,
        set(layer_ids),
        target_p99,
    )
    outputs = []
    for item in layer_logits:
        logits = item["logits"]
        reference = torch.softmax(logits, dim=-1)
        encrypted = crypten.cryptensor(logits, src=0)
        cases = build_cases(scales, scaled_iters, scaled_clip_iters)
        outputs.append(
            {
                "layer": item["layer"],
                "shape": tuple(logits.shape),
                "logit_min": item["min"],
                "logit_max": item["max"],
                "logit_std": item["std"],
                "results": [
                    (name, measure(encrypted, reference, fn, repeats))
                    for name, fn in cases
                ],
            }
        )
    return {"rank": comm.get().get_rank(), "outputs": outputs}


def estimate_time(compute_s, comm_bytes, rounds, bandwidth_bps, latency_s):
    return compute_s + 2 * comm_bytes / (bandwidth_bps / 8) + rounds * latency_s


def summarize(rank_outputs):
    for layer_pos, first in enumerate(rank_outputs[0]["outputs"]):
        print(
            f"\nlayer={first['layer']} shape={first['shape']} "
            f"logit_min={first['logit_min']:.4f} "
            f"logit_max={first['logit_max']:.4f} "
            f"logit_std={first['logit_std']:.4f}"
        )
        print(
            f"{'case':<18} {'comp_ms':>9} {'comm_MB':>9} {'rounds':>7} "
            f"{'LAN_ms':>9} {'WAN_ms':>9} {'max_abs':>9} {'mean_abs':>9}"
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
            wan = estimate_time(compute, comm_bytes, rounds, WAN_BPS, WAN_LATENCY_S)
            print(
                f"{name:<18} {compute * 1000:9.3f} {comm_bytes / 1_000_000:9.4f} "
                f"{rounds:7.1f} {lan * 1000:9.3f} {wan * 1000:9.3f} "
                f"{max(item['max_abs'] for item in items):9.5f} "
                f"{max(item['mean_abs'] for item in items):9.5f}"
            )


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-p99", type=float, default=4.0)
    parser.add_argument("--scales", type=int, nargs="+", default=[2])
    parser.add_argument("--scaled-iters", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--scaled-clip-iters", type=int, nargs="+", default=[])
    args = parser.parse_args()
    outputs = run_rank(
        args.model_path,
        args.dataset_path,
        args.max_length,
        args.batch_size,
        args.offset,
        args.layers,
        args.repeats,
        args.target_p99,
        sorted(set(args.scales)),
        sorted(set(args.scaled_iters)),
        sorted(set(args.scaled_clip_iters)),
    )
    summarize(outputs)


if __name__ == "__main__":
    main()
