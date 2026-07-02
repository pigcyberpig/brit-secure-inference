import argparse
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import crypten
import crypten.communicator as comm


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


def split_heads(x, num_heads):
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)


def get_masked_logits(model_path, dataset_path, max_length, offset, layers):
    encoded = load_padded_sample(dataset_path, model_path, max_length, offset)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    bert = model.bert
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
            attn = layer.attention.self
            query = split_heads(attn.query(hidden), attn.num_attention_heads)
            key = split_heads(attn.key(hidden), attn.num_attention_heads)
            logits = torch.matmul(query, key.transpose(-1, -2))
            logits = logits / math.sqrt(attn.attention_head_size)
            logits = logits + attention_mask
            if layer_idx in layers:
                outputs.append(
                    {
                        "layer": layer_idx,
                        "logits": logits.reshape(-1, max_length).contiguous(),
                        "valid_len": int(encoded["attention_mask"].sum().item()),
                    }
                )
            hidden = layer(hidden, attention_mask=attention_mask)[0]
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


def ode_softmax(tensor, dim, iters, clip):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": clip,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        return tensor.softmax(dim=dim)


def scaled_k_softmax(tensor, dim, scale, iters):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / scale
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        probs = scaled.softmax(dim=dim)
    powered = pow_int(probs, scale)
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return powered * powered.sum(dim=dim, keepdim=True).reciprocal()


def bumblebee_softmax(tensor, dim, taylor_iters, texp):
    # BumbleBee (NDSS'25) §5.B: max-subtract + clip[Texp,0] + limit-exp(depth=n) + reciprocal.
    # The paper's "Taylor degree-n" = limit method exp(x)=[exp(x/2^n)]^(2^n) with a
    # degree-2 inner Taylor; that is exactly crypten's tensor.exp() with exp_iterations=n.
    # Paper uses n=6 with fixed-point f=18 -> Texp=-13; CrypTen precision_bits=16
    # (configs/default.yaml) -> exp(Texp)<2^-16 -> Texp=-12.
    maximum = tensor.max(dim=dim, keepdim=True)[0]
    neg = tensor - maximum  # all exp inputs <= 0
    # clip lower bound to texp: exp(texp) is already < 1 LSB, no separate exp-after branch.
    clipped = neg + (texp - neg).relu()  # lift values < texp up to texp
    with crypten.cfg.temp_override({"functions.exp_iterations": taylor_iters}):
        num = clipped.exp()
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return num * num.sum(dim=dim, keepdim=True).reciprocal()


def bolt_recip_softmax(tensor, dim, exp_iters):
    # BOLT (S&P'24) §5.4 framework-aligned variant: max-subtract + exp + reciprocal.
    # NOTE: BOLT's actual exp is I-BERT integer/fraction decomposition
    # (exp(p) ~= 0.3585(p+1.353)^2+0.344 for p in (-ln2,0], shift-by-z), which is NOT
    # reproduced here by decision. Without that decomposition this is equivalent to
    # crypten's softmax_method="reciprocal"; kept as a max-subtract+exp+recip backbone.
    maximum = tensor.max(dim=dim, keepdim=True)[0]
    with crypten.cfg.temp_override({"functions.exp_iterations": exp_iters}):
        num = (tensor - maximum).exp()
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return num * num.sum(dim=dim, keepdim=True).reciprocal()


def measure(encrypted, reference, fn, repeats, valid_len):
    records = []
    last = None
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        result = fn(encrypted)
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
    seq_len = reference.size(1)
    num_heads = reference.size(0) // seq_len
    valid_rows = []
    for head_idx in range(num_heads):
        base = head_idx * seq_len
        valid_rows.extend(range(base, base + valid_len))
    valid_rows = torch.tensor(valid_rows, dtype=torch.long)
    valid_last = last.index_select(0, valid_rows)[:, :valid_len]
    valid_ref = reference.index_select(0, valid_rows)[:, :valid_len]
    valid_diff = (valid_last - valid_ref).abs()
    valid_last_renorm = valid_last / valid_last.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    valid_ref_renorm = valid_ref / valid_ref.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    valid_renorm_diff = (valid_last_renorm - valid_ref_renorm).abs()
    masked_last = last.index_select(0, valid_rows)[:, valid_len:]
    masked_ref = reference.index_select(0, valid_rows)[:, valid_len:]
    masked_diff = (masked_last - masked_ref).abs()
    masked_mass = masked_last.sum(dim=-1)
    avg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    avg["compute"] = max(avg["wall"] - avg["comm_time"], 0.0)
    avg["full_max_abs"] = diff.max().item()
    avg["full_mean_abs"] = diff.mean().item()
    avg["full_row_sum"] = (last.sum(dim=-1) - 1).abs().max().item()
    avg["valid_max_abs"] = valid_diff.max().item()
    avg["valid_mean_abs"] = valid_diff.mean().item()
    avg["valid_row_sum"] = (valid_last.sum(dim=-1) - valid_ref.sum(dim=-1)).abs().max().item()
    avg["valid_renorm_max_abs"] = valid_renorm_diff.max().item()
    avg["valid_renorm_mean_abs"] = valid_renorm_diff.mean().item()
    avg["masked_max_abs"] = masked_diff.max().item() if masked_diff.numel() else 0.0
    avg["masked_mean_abs"] = masked_diff.mean().item() if masked_diff.numel() else 0.0
    avg["masked_mass_max"] = masked_mass.max().item() if masked_mass.numel() else 0.0
    avg["masked_mass_mean"] = masked_mass.mean().item() if masked_mass.numel() else 0.0
    return avg


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(model_path, dataset_path, max_length, offset, layer_ids, repeats,
             synthetic_heads=None):
    device = resolve_device()
    outputs = []
    cases = [
        ("ode_clip_i16", lambda x: ode_softmax(x, -1, 16, True)),
        ("scaled_k2_i10", lambda x: scaled_k_softmax(x, -1, 2, 10)),
        ("scaled_k2_i12", lambda x: scaled_k_softmax(x, -1, 2, 12)),
        ("scaled_k2_i16", lambda x: scaled_k_softmax(x, -1, 2, 16)),
        ("bumblebee_i6", lambda x: bumblebee_softmax(x, -1, 6, -12)),
        ("bolt_recip_i8", lambda x: bolt_recip_softmax(x, -1, 8)),
    ]
    if synthetic_heads is not None:
        # Synthetic per-layer softmax of shape (heads*seq, seq): mimics one
        # attention layer's full multi-head softmax. Comm/rounds are
        # shape-determined, so random values are equivalent to real logits.
        heads = synthetic_heads
        logits = torch.randn(heads * max_length, max_length)
        reference = torch.softmax(logits, dim=-1)
        encrypted = crypten.cryptensor(logits, src=0).to(device)
        outputs.append(
            {
                "layer": 0,
                "shape": tuple(logits.shape),
                "valid_len": max_length,
                "results": [
                    (name, measure(encrypted, reference, fn, repeats, max_length))
                    for name, fn in cases
                ],
            }
        )
        return {"rank": comm.get().get_rank(), "outputs": outputs}
    for item in get_masked_logits(
        model_path, dataset_path, max_length, offset, set(layer_ids)
    ):
        # Per-head 128x128 input: take head 0 (first max_length rows of the
        # flattened 12-head x 128 layout). Comm/rounds are shape-determined,
        # so a single head is representative for the per-operator comparison.
        logits = item["logits"][:max_length]
        reference = torch.softmax(logits, dim=-1)
        encrypted = crypten.cryptensor(logits, src=0).to(device)
        outputs.append(
            {
                "layer": item["layer"],
                "shape": tuple(logits.shape),
                "valid_len": item["valid_len"],
                "results": [
                    (name, measure(encrypted, reference, fn, repeats, item["valid_len"]))
                    for name, fn in cases
                ],
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
            f"{'case':<18} {'comp_ms':>9} {'comm_MB':>9} {'rounds':>7} "
            f"{'LAN_ms':>9} {'WAN4_ms':>9} {'WAN80_ms':>10} "
            f"{'renorm_max':>11} {'renorm_mean':>11} "
            f"{'masked_mass':>12}"
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
                    "valid_renorm_max_abs": max(
                        item["valid_renorm_max_abs"] for item in items
                    ),
                    "valid_renorm_mean_abs": max(
                        item["valid_renorm_mean_abs"] for item in items
                    ),
                    "masked_mass_max": max(item["masked_mass_max"] for item in items),
                }
            )
            print(
                f"{name:<18} {compute * 1000:9.3f} {comm_bytes / 1_000_000:9.4f} "
                f"{rounds:7.1f} {lan * 1000:9.3f} {wan_mid * 1000:9.3f} "
                f"{wan_hard * 1000:10.3f} "
                f"{max(item['valid_renorm_max_abs'] for item in items):11.5f} "
                f"{max(item['valid_renorm_mean_abs'] for item in items):11.5f} "
                f"{max(item['masked_mass_max'] for item in items):12.5f}"
            )
            print(
                f"{'':<18} {'':>9} {'':>9} {'':>7} {'':>9} {'':>9} "
                f"valid_max={max(item['valid_max_abs'] for item in items):.5f} "
                f"valid_mean={max(item['valid_mean_abs'] for item in items):.5f} "
                f"masked_max={max(item['masked_max_abs'] for item in items):.5f} "
                f"masked_mean={max(item['masked_mean_abs'] for item in items):.5f}"
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
    parser.add_argument(
        "--synthetic-heads", type=int, default=None,
        help="If set, use synthetic random logits of (heads*seq, seq) and skip "
             "model loading. Used to test a full multi-head attention layer "
             "(e.g. 12 for BERT-base, 16 for BERT-large).",
    )
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    outputs = run_rank(
        args.model_path,
        args.dataset_path,
        args.max_length,
        args.offset,
        args.layers,
        args.repeats,
        args.synthetic_heads,
    )
    summary = summarize(outputs)
    if args.json_output:
        with open(args.json_output, "w") as handle:
            json.dump({"results": summary}, handle, indent=2)


if __name__ == "__main__":
    main()
