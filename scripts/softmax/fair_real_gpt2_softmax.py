import argparse
import math
import os
import time

import torch
from transformers import AutoModel, AutoTokenizer

import crypten
import crypten.communicator as comm


LAN_BPS = 1_000_000_000
LAN_LATENCY_S = 0.0005
WAN_BPS = 400_000_000
WAN_LATENCY_S = 0.004


TEXTS = [
    "Privacy preserving transformer inference needs accurate attention softmax.",
    "Secure two party computation often trades communication rounds for bandwidth.",
    "The scaled ODE softmax experiment should use real model attention logits.",
    "Large tensors are batched together in practical transformer inference.",
    "A fair comparison estimates LAN and WAN time from communication statistics.",
    "Cryptographic protocols make nonlinear functions expensive to evaluate.",
    "Attention logits can have different ranges across layers and heads.",
    "This benchmark extracts GPT2 query key scores before softmax.",
]


def split_heads(x, num_heads):
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)


def get_gpt2_logits(model_path, seq_len, batch_size, layers, target_p99):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [TEXTS[i % len(TEXTS)] for i in range(batch_size)]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )

    with torch.no_grad():
        hidden = model.wte(encoded["input_ids"]) + model.wpe(
            torch.arange(seq_len).unsqueeze(0)
        )
        logits_by_layer = []
        for layer_idx, block in enumerate(model.h):
            attn_input = block.ln_1(hidden)
            qkv = block.attn.c_attn(attn_input)
            query, key, _ = qkv.split(model.config.n_embd, dim=2)
            query = split_heads(query, model.config.n_head)
            key = split_heads(key, model.config.n_head)
            logits = torch.matmul(query, key.transpose(-1, -2))
            logits = logits / math.sqrt(query.size(-1))
            logits = logits.reshape(-1, seq_len)
            if target_p99 is not None:
                centered = logits - logits.mean(dim=-1, keepdim=True)
                p99 = torch.quantile(centered.abs().reshape(-1), 0.99).clamp_min(1e-6)
                logits = centered * (target_p99 / p99)
            if layer_idx in layers:
                logits_by_layer.append((layer_idx, logits.contiguous()))
            outputs = block(hidden, attention_mask=None, use_cache=False)
            hidden = outputs[0]
    return logits_by_layer


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


def scaled_k2_softmax(tensor, dim, iters):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / 2
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        probs = scaled.softmax(dim=dim)
    probs2 = probs * probs
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return probs2 * probs2.sum(dim=dim, keepdim=True).reciprocal()


def ode_softmax(tensor, dim, clip):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": clip,
            "functions.softmax_ode_iter_num": 16,
        }
    ):
        return tensor.softmax(dim=dim)


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
def run_rank(model_path, seq_len, batch_size, layer_ids, repeats, target_p99):
    logits_by_layer = get_gpt2_logits(
        model_path, seq_len, batch_size, set(layer_ids), target_p99
    )
    outputs = []
    for layer_idx, logits in logits_by_layer:
        reference = torch.softmax(logits, dim=-1)
        encrypted = crypten.cryptensor(logits, src=0)
        cases = [
            ("ode_clip_i16", lambda x: ode_softmax(x, -1, True)),
            ("ode_no_clip_i16", lambda x: ode_softmax(x, -1, False)),
            ("scaled_k2_i8", lambda x: scaled_k2_softmax(x, -1, 8)),
            ("scaled_k2_i4", lambda x: scaled_k2_softmax(x, -1, 4)),
        ]
        outputs.append(
            {
                "layer": layer_idx,
                "shape": tuple(logits.shape),
                "logit_min": logits.min().item(),
                "logit_max": logits.max().item(),
                "logit_std": logits.std().item(),
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
    first = rank_outputs[0]["outputs"]
    for layer_pos, layer_output in enumerate(first):
        print(
            f"\nlayer={layer_output['layer']} shape={layer_output['shape']} "
            f"logit_min={layer_output['logit_min']:.4f} "
            f"logit_max={layer_output['logit_max']:.4f} "
            f"logit_std={layer_output['logit_std']:.4f}"
        )
        print(
            f"{'case':<16} {'comp_ms':>9} {'comm_MB':>9} {'rounds':>7} "
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
            max_abs = max(item["max_abs"] for item in items)
            mean_abs = max(item["mean_abs"] for item in items)
            print(
                f"{name:<16} {compute * 1000:9.3f} {comm_bytes / 1_000_000:9.4f} "
                f"{rounds:7.1f} {lan * 1000:9.3f} {wan * 1000:9.3f} "
                f"{max_abs:9.5f} {mean_abs:9.5f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("GPT2_MODEL", "gpt2"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-p99", type=float, default=4.0)
    args = parser.parse_args()

    outputs = run_rank(
        args.model_path,
        args.seq_len,
        args.batch_size,
        args.layers,
        args.repeats,
        args.target_p99,
    )
    summarize(outputs)


if __name__ == "__main__":
    main()
