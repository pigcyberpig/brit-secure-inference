import argparse
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = os.path.join(os.environ.get("DATA_ROOT", ""), "bert-base-cased-sst2")
DEFAULT_VALIDATION = (
    "artifacts/experiment/sst2_scaled_tradeoff_20260524/"
    "half_subset_seed20260524/validation.parquet"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure empirical leakage of revealing W = X * Z^2 for LayerNorm variance X."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--validation_file", default=DEFAULT_VALIDATION)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--x_bins", type=int, default=64)
    parser.add_argument("--w_bins", type=int, default=256)
    parser.add_argument("--sim_samples", type=int, default=200000)
    parser.add_argument("--z_max", type=int, nargs="+", default=[15, 63, 255, 1023])
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use cuda only when CUDA_VISIBLE_DEVICES has been constrained to GPU0.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_validation(validation_file):
    extension = Path(validation_file).suffix.lstrip(".")
    if extension != "parquet":
        raise ValueError(f"Only parquet validation files are supported, got {validation_file}")
    return load_dataset("parquet", data_files={"validation": validation_file}, split="validation")


def collect_layernorm_variances(args, device):
    dataset = load_validation(args.validation_file)
    n_samples = min(args.samples, len(dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=True
    ).to(device)
    model.eval()

    current_mask = {"value": None}
    by_module = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0].detach().float()
            if x.dim() != 3:
                return
            var = x.var(dim=-1, unbiased=False)
            mask = current_mask["value"]
            if mask is not None and mask.shape == var.shape:
                var = var[mask.bool()]
            by_module[name].append(var.detach().cpu())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    labels = []
    lengths = []
    start = time.time()
    try:
        with torch.no_grad():
            for offset in range(0, n_samples, args.batch_size):
                batch_rows = dataset.select(range(offset, min(offset + args.batch_size, n_samples)))
                sentences = batch_rows["sentence"]
                encoded = tokenizer(
                    sentences,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                current_mask["value"] = encoded["attention_mask"].to(device)
                labels.extend(int(v) for v in batch_rows["label"])
                lengths.extend(int(v) for v in encoded["attention_mask"].sum(dim=1).tolist())
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model(**encoded)
    finally:
        for hook in hooks:
            hook.remove()

    module_arrays = {}
    for name, chunks in by_module.items():
        if chunks:
            module_arrays[name] = torch.cat(chunks).numpy().astype(np.float64)
    all_x = np.concatenate(list(module_arrays.values()))
    all_x = all_x[np.isfinite(all_x) & (all_x > 0)]

    meta = {
        "samples_requested": args.samples,
        "samples_used": n_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "valid_token_count": int(sum(lengths)),
        "layernorm_modules": sorted(module_arrays),
        "elapsed_s": time.time() - start,
    }
    return all_x, module_arrays, np.asarray(labels), np.asarray(lengths), meta


def quantile_summary(values):
    qs = [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    return {f"p{q * 100:g}": float(np.quantile(values, q)) for q in qs}


def make_quantile_edges(values, bins):
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("Not enough unique values to build quantile bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def digitize(values, edges):
    return np.searchsorted(edges[1:-1], values, side="right")


def entropy_bits(prob):
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def auc_rank(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def histogram_tv(a, b, bins):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b, bins=edges, density=False)
    pa = hist_a / max(1, hist_a.sum())
    pb = hist_b / max(1, hist_b.sum())
    return float(0.5 * np.abs(pa - pb).sum())


def bayes_leakage_metrics(x_values, z_max, args, rng):
    x_edges = make_quantile_edges(x_values, args.x_bins)
    n_x_bins = len(x_edges) - 1
    prior_x = rng.choice(x_values, size=args.sim_samples, replace=True)
    z = rng.integers(1, z_max + 1, size=args.sim_samples)
    log_w = np.log(prior_x) + 2.0 * np.log(z)
    x_bin = digitize(prior_x, x_edges)

    split = args.sim_samples // 2
    train_w = log_w[:split]
    train_x = x_bin[:split]
    test_w = log_w[split:]
    test_x = x_bin[split:]

    w_edges = np.linspace(float(train_w.min()), float(train_w.max()), args.w_bins + 1)
    w_edges[0] = -np.inf
    w_edges[-1] = np.inf
    train_w_bin = digitize(train_w, w_edges)
    test_w_bin = digitize(test_w, w_edges)
    n_w_bins = len(w_edges) - 1

    joint = np.zeros((n_x_bins, n_w_bins), dtype=np.float64)
    np.add.at(joint, (train_x, train_w_bin), 1.0)
    joint_smoothed = joint + 1e-9
    joint_prob = joint_smoothed / joint_smoothed.sum()
    px = joint_prob.sum(axis=1)
    pw = joint_prob.sum(axis=0)
    expected = px[:, None] * pw[None, :]
    mi = float((joint_prob * np.log2(joint_prob / expected)).sum())
    hx = entropy_bits(px)

    likelihood = joint_smoothed / joint_smoothed.sum(axis=1, keepdims=True)
    posterior = likelihood[:, test_w_bin] * px[:, None]
    posterior /= posterior.sum(axis=0, keepdims=True)
    pred = posterior.argmax(axis=0)
    top1 = float((pred == test_x).mean())
    top5_idx = np.argpartition(posterior, -min(5, n_x_bins), axis=0)[-min(5, n_x_bins) :, :]
    top5 = float(np.any(top5_idx == test_x[None, :], axis=0).mean())
    posterior_entropy = float(np.mean(-np.sum(posterior * np.log2(np.clip(posterior, 1e-300, 1)), axis=0)))

    centers = []
    for i in range(n_x_bins):
        lo = x_edges[i]
        hi = x_edges[i + 1]
        if np.isneginf(lo):
            lo = np.min(x_values)
        if np.isposinf(hi):
            hi = np.max(x_values)
        centers.append(math.sqrt(max(lo, 1e-300) * max(hi, 1e-300)))
    centers = np.asarray(centers)
    rel_err = np.abs(centers[pred] - prior_x[split:]) / np.maximum(prior_x[split:], 1e-300)

    return {
        "z_max": z_max,
        "x_bins": n_x_bins,
        "w_bins": n_w_bins,
        "mutual_information_bits": mi,
        "x_entropy_bits": hx,
        "normalized_mi": mi / hx if hx > 0 else float("nan"),
        "posterior_entropy_bits": posterior_entropy,
        "top1_xbin_accuracy": top1,
        "top5_xbin_accuracy": top5,
        "map_relative_error_median": float(np.median(rel_err)),
        "map_relative_error_p90": float(np.quantile(rel_err, 0.9)),
    }


def pairwise_metrics(x_values, z_max, rng, pair_count=40000):
    q_pairs = [(0.01, 0.05), (0.25, 0.5), (0.5, 0.75), (0.95, 0.99)]
    out = []
    for qa, qb in q_pairs:
        xa = float(np.quantile(x_values, qa))
        xb = float(np.quantile(x_values, qb))
        z0 = rng.integers(1, z_max + 1, size=pair_count)
        z1 = rng.integers(1, z_max + 1, size=pair_count)
        w0 = np.log(xa) + 2.0 * np.log(z0)
        w1 = np.log(xb) + 2.0 * np.log(z1)
        labels = np.concatenate([np.zeros(pair_count), np.ones(pair_count)])
        scores = np.concatenate([w0, w1])
        out.append(
            {
                "q_low": qa,
                "q_high": qb,
                "x_low": xa,
                "x_high": xb,
                "auc_logw_threshold": auc_rank(labels, scores),
                "tv_logw_hist": histogram_tv(w0, w1, bins=512),
            }
        )
    return out


def write_markdown(path, metrics):
    lines = [
        "# LayerNorm Mask Leakage Experiment",
        "",
        "## Variance Summary",
        "",
        f"- Samples used: `{metrics['collection']['samples_used']}`",
        f"- Valid token count: `{metrics['collection']['valid_token_count']}`",
        f"- LayerNorm variance values: `{metrics['variance_count']}`",
        f"- Variance p1 / median / p99: `{metrics['variance_summary']['p1']:.6g}` / "
        f"`{metrics['variance_summary']['p50']:.6g}` / `{metrics['variance_summary']['p99']:.6g}`",
        "",
        "## Leakage Metrics",
        "",
        "| z_max | MI bits | normalized MI | posterior H bits | top1 x-bin | top5 x-bin | median rel err |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["leakage"]:
        lines.append(
            f"| {item['z_max']} | {item['mutual_information_bits']:.4f} | "
            f"{item['normalized_mi']:.4f} | {item['posterior_entropy_bits']:.4f} | "
            f"{item['top1_xbin_accuracy']:.4f} | {item['top5_xbin_accuracy']:.4f} | "
            f"{item['map_relative_error_median']:.4f} |"
        )
    lines.extend(["", "## Pairwise Distinguishability", ""])
    for z_item in metrics["pairwise"]:
        lines.append(f"### z_max={z_item['z_max']}")
        lines.append("")
        lines.append("| quantiles | x_low | x_high | AUC(log W) | TV(hist) |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in z_item["pairs"]:
            lines.append(
                f"| p{row['q_low'] * 100:g} vs p{row['q_high'] * 100:g} | "
                f"{row['x_low']:.6g} | {row['x_high']:.6g} | "
                f"{row['auc_logw_threshold']:.4f} | {row['tv_logw_hist']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This is an empirical leakage evaluation for a TTP-generated positive mask. "
            "It does not prove ideal-real security; it measures how much the revealed "
            "`W = X * Z^2` helps infer binned LayerNorm variance under the observed SST-2/BERT prior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    x_values, by_module, labels, lengths, collection = collect_layernorm_variances(args, device)
    np.save(out_dir / "layernorm_variances.npy", x_values)
    module_summary = {
        name: {"count": int(values.size), "summary": quantile_summary(values)}
        for name, values in sorted(by_module.items())
    }

    leakage = [bayes_leakage_metrics(x_values, z, args, rng) for z in args.z_max]
    pairwise = [
        {"z_max": z, "pairs": pairwise_metrics(x_values, z, rng)} for z in args.z_max
    ]

    metrics = {
        "run_id": out_dir.name,
        "seed": args.seed,
        "device": str(device),
        "model_path": args.model_path,
        "validation_file": args.validation_file,
        "variance_count": int(x_values.size),
        "variance_summary": quantile_summary(x_values),
        "module_summary": module_summary,
        "collection": collection,
        "leakage": leakage,
        "pairwise": pairwise,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown(out_dir / "metrics.md", metrics)

    run_manifest = {
        "run_id": out_dir.name,
        "script": "scripts/layernorm/layernorm_mask_leakage.py",
        "argv": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    "layernorm_variances.npy",
                    "metrics.json",
                    "metrics.md",
                    "run_manifest.json",
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "variance_count": int(x_values.size)}, indent=2))


if __name__ == "__main__":
    main()
