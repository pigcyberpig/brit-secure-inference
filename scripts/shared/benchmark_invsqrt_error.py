"""Inverse-sqrt sub-protocol error benchmark: NR (Newton-Raphson) vs MLFormer.

Isolates the 1/sqrt(x+eps) sub-protocol from full LayerNorm. The input x is the
variance (sigma^2) that LayerNorm feeds into inv_sqrt. We measure on two input
sets:
  1. Real BERT-base LayerNorm variances (seq=128, hidden=768, layers 0/5/11).
  2. A synthetic sweep over a configurable range to expose where NR diverges.

For each input we compare the secure inv_sqrt output (decrypted) against the
plaintext 1/sqrt(x+eps) and report max/mean abs and relative error.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import crypten
import crypten.communicator as comm

EPS = 1e-5  # matches _inv_sqrt_mlformer default


def resolve_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return "cuda:0"
    return "cpu"


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


def get_real_variances(model_path, dataset_path, max_length, offset, layers):
    """Return per-layer variance vectors (sigma^2 over hidden dim) feeding inv_sqrt."""
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
                var = hidden.var(dim=-1, keepdim=False)  # (1, seq) -> flatten
                outputs.append(
                    {"layer": layer_idx, "var": var.reshape(-1).contiguous()}
                )
            hidden = layer(hidden, attention_mask=attention_mask)[0]
    return outputs, hidden_size


def synthetic_sweep(n=4096, lo=0.01, hi=10.0, mode="random_loguniform"):
    if mode == "logspace":
        return torch.logspace(math.log10(lo), math.log10(hi), steps=n)
    if mode == "random_loguniform":
        mag = torch.rand(n) * (math.log10(hi) - math.log10(lo)) + math.log10(lo)
        return 10.0 ** mag
    raise ValueError(f"unsupported sweep mode: {mode}")


def make_key_xs():
    return torch.tensor(
        [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        dtype=torch.float32,
    )


def measure_set(name, x_plain, device, include_curve=False):
    """Run NR and MLFormer inv_sqrt on encrypted x, report error vs plaintext."""
    x_plain = x_plain.to(device)
    reference = 1.0 / torch.sqrt(x_plain + EPS)
    enc = crypten.cryptensor(x_plain, src=0).to(device)
    out = {}
    curve = {}
    for method in ("NR", "MLFormer"):
        comm.get().barrier()
        crypten.reset_communication_stats()
        with crypten.cfg.temp_override({"functions.sqrt_method": method}):
            y = enc.inv_sqrt()
        y_plain = y.get_plain_text().cpu()
        ref = reference.cpu()
        abs_err = (y_plain - ref).abs()
        rel_err = (abs_err / ref.abs().clamp_min(1e-12))
        out[method] = {
            "max_abs": abs_err.max().item(),
            "mean_abs": abs_err.mean().item(),
            "max_rel": rel_err.max().item(),
            "mean_rel": rel_err.mean().item(),
        }
        if include_curve:
            curve[method] = {
                "abs_err": abs_err.tolist(),
                "rel_err": rel_err.tolist(),
            }
    if include_curve:
        return {
            "summary": out,
            "curve": {
                "x": x_plain.cpu().tolist(),
                "reference": reference.cpu().tolist(),
                "methods": curve,
            },
        }
    return out


def attach_key_points(curve_payload, key_xs):
    x_vals = curve_payload["curve"]["x"]
    nr_abs = curve_payload["curve"]["methods"]["NR"]["abs_err"]
    nr_rel = curve_payload["curve"]["methods"]["NR"]["rel_err"]
    mlf_abs = curve_payload["curve"]["methods"]["MLFormer"]["abs_err"]
    mlf_rel = curve_payload["curve"]["methods"]["MLFormer"]["rel_err"]
    key_points = []
    for x_target in key_xs.tolist():
        idx = min(range(len(x_vals)), key=lambda i: abs(x_vals[i] - x_target))
        key_points.append(
            {
                "target_x": x_target,
                "x": x_vals[idx],
                "NR": {"abs_err": nr_abs[idx], "rel_err": nr_rel[idx]},
                "MLFormer": {"abs_err": mlf_abs[idx], "rel_err": mlf_rel[idx]},
            }
        )
    curve_payload["key_points"] = key_points
    return curve_payload


def write_keypoint_csv(path, key_points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("target_x,x,method,abs_err,rel_err\n")
        for row in key_points:
            for method in ("NR", "MLFormer"):
                f.write(
                    f"{row['target_x']},{row['x']},{method},"
                    f"{row[method]['abs_err']},{row[method]['rel_err']}\n"
                )


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(
    model_path,
    dataset_path,
    max_length,
    offset,
    layer_ids,
    skip_real,
    sweep_lo,
    sweep_hi,
    sweep_points,
    sweep_mode,
):
    device = resolve_device()
    results = {}
    # 1. real variances
    real_results = []
    if not skip_real:
        real, _ = get_real_variances(
            model_path, dataset_path, max_length, offset, set(layer_ids)
        )
        for item in real:
            v = item["var"].float()
            err = measure_set(f"layer{item['layer']}", v, device)
            real_results.append({"layer": item["layer"], "n": v.numel(), "error": err})
    # 2. synthetic sweep
    sweep = synthetic_sweep(sweep_points, sweep_lo, sweep_hi, sweep_mode)
    sweep_err = measure_set("sweep", sweep.float(), device, include_curve=True)
    key_xs = make_key_xs()
    if sweep_lo <= float(key_xs.min()) and sweep_hi >= float(key_xs.max()):
        sweep_err = attach_key_points(sweep_err, key_xs)
    results = {
        "real": real_results,
        "sweep": {
            "n": sweep.numel(),
            "lo": sweep_lo,
            "hi": sweep_hi,
            "mode": sweep_mode,
            **sweep_err,
        },
    }
    return {"rank": comm.get().get_rank(), "results": results}


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
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--sweep-lo", type=float, default=0.01)
    parser.add_argument("--sweep-hi", type=float, default=10.0)
    parser.add_argument("--sweep-points", type=int, default=4096)
    parser.add_argument(
        "--sweep-mode",
        choices=["random_loguniform", "logspace"],
        default="random_loguniform",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--keypoints-csv-output", default=None)
    args = parser.parse_args()
    outputs = run_rank(
        args.model_path,
        args.dataset_path,
        args.max_length,
        args.offset,
        args.layers,
        args.skip_real,
        args.sweep_lo,
        args.sweep_hi,
        args.sweep_points,
        args.sweep_mode,
    )
    # take rank 0
    res = outputs[0]["results"]
    print("\n=== Inverse-sqrt sub-protocol error (1/sqrt(x+eps)) ===\n")
    if res["real"]:
        print("Real BERT LayerNorm variances:")
        print(f"{'layer':<6}{'method':<10}{'max_abs':>12}{'mean_abs':>12}{'max_rel':>12}{'mean_rel':>12}")
        for item in res["real"]:
            for method, e in item["error"].items():
                print(
                    f"{item['layer']:<6}{method:<10}{e['max_abs']:12.5f}"
                    f"{e['mean_abs']:12.6f}{e['max_rel']:12.4f}{e['mean_rel']:12.4f}"
                )
    print(f"\nSynthetic {args.sweep_mode} sweep [{args.sweep_lo}, {args.sweep_hi}]:")
    print(f"{'method':<10}{'max_abs':>12}{'mean_abs':>12}{'max_rel':>12}{'mean_rel':>12}")
    for method, e in res["sweep"]["summary"].items():
        print(f"{method:<10}{e['max_abs']:12.5f}{e['mean_abs']:12.6f}{e['max_rel']:12.4f}{e['mean_rel']:12.4f}")
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(res, f, indent=2)
    if args.keypoints_csv_output and "key_points" in res["sweep"]:
        write_keypoint_csv(args.keypoints_csv_output, res["sweep"]["key_points"])


if __name__ == "__main__":
    main()
