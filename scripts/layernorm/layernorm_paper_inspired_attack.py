import argparse
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = os.path.join(os.environ.get("DATA_ROOT", ""), "bert-base-cased-sst2")
DEFAULT_VALIDATION = (
    "artifacts/experiment/softmax/sst2_scaled_tradeoff_20260524/"
    "half_subset_seed20260524/validation.parquet"
)

LEAK_CONFIG = {
    "name": "frac_5M_S50",
    "candidates": 5_000_000,
    "scale": 50.0,
    "eps": 1e-5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper-inspired LayerNorm privacy attack. Measures token reconstruction "
            "from the protocol's actually revealed log W."
        )
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--validation_file", default=DEFAULT_VALIDATION)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=436)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_tokens", type=int, default=100)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument(
        "--share_dims",
        type=int,
        default=2048,
        help=(
            "Dimension of the simulated single-party activation-share view. "
            "Use 0 for module_count * hidden_size."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_validation(path):
    return load_dataset("parquet", data_files={"validation": path}, split="validation")


def collect_records(args, device):
    dataset = load_validation(args.validation_file)
    n_samples = min(args.samples, len(dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=True
    ).to(device)
    model.eval()

    module_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)
    ]
    if not module_names:
        raise RuntimeError("No LayerNorm modules found")
    module_to_idx = {name: idx for idx, name in enumerate(module_names)}

    sample_vars = [[None for _ in module_names] for _ in range(n_samples)]
    sample_token_ids = [None for _ in range(n_samples)]
    sample_lengths = np.zeros(n_samples, dtype=np.int64)
    sample_labels = np.zeros(n_samples, dtype=np.int64)
    sample_logits = np.zeros((n_samples, model.config.num_labels), dtype=np.float64)
    current = {"sample_ids": None, "mask": None}
    hooks = []

    def make_hook(name):
        module_idx = module_to_idx[name]

        def hook(_module, inputs):
            x = inputs[0].detach().float()
            if x.dim() != 3:
                return
            mask = current["mask"]
            sample_ids = current["sample_ids"]
            if mask is None or sample_ids is None:
                return
            mask_np = mask.detach().cpu().numpy().astype(bool)
            var = x.var(dim=-1, unbiased=False).detach().cpu().numpy()
            for row_idx, sample_id in enumerate(sample_ids):
                valid = mask_np[row_idx]
                sample_vars[sample_id][module_idx] = var[row_idx][valid].astype(np.float64)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    start = time.time()
    try:
        with torch.no_grad():
            for offset in range(0, n_samples, args.batch_size):
                end = min(offset + args.batch_size, n_samples)
                rows = dataset.select(range(offset, end))
                encoded = tokenizer(
                    rows["sentence"],
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                mask = encoded["attention_mask"]
                current["sample_ids"] = list(range(offset, end))
                current["mask"] = mask.to(device)
                for local_idx, sample_id in enumerate(range(offset, end)):
                    valid = mask[local_idx].bool()
                    sample_token_ids[sample_id] = encoded["input_ids"][local_idx][
                        valid
                    ].cpu().numpy().astype(np.int64)
                    sample_lengths[sample_id] = int(valid.sum().item())
                    sample_labels[sample_id] = int(rows["label"][local_idx])
                outputs = model(**{k: v.to(device) for k, v in encoded.items()})
                sample_logits[offset:end] = outputs.logits.detach().cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()

    special_ids = set(tokenizer.all_special_ids)
    records = []
    for sample_id in range(n_samples):
        token_ids = sample_token_ids[sample_id]
        if token_ids is None:
            continue
        variances = np.stack(sample_vars[sample_id], axis=1)
        for pos, token_id in enumerate(token_ids.tolist()):
            if token_id in special_ids:
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "pos": pos,
                    "length": int(sample_lengths[sample_id]),
                    "label": int(sample_labels[sample_id]),
                    "token_id": int(token_id),
                    "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                    "variance": variances[pos].astype(np.float64),
                    "logits": sample_logits[sample_id].astype(np.float64),
                }
            )

    meta = {
        "samples_used": n_samples,
        "content_token_count": len(records),
        "module_count": len(module_names),
        "module_names": module_names,
        "hidden_size": int(model.config.hidden_size),
        "elapsed_s": time.time() - start,
    }
    return records, meta


def select_rows(records, top_tokens, min_count):
    counts = Counter(row["token_id"] for row in records)
    selected = [
        token_id for token_id, count in counts.most_common() if count >= min_count
    ][:top_tokens]
    selected_set = set(selected)
    rows = [row for row in records if row["token_id"] in selected_set]
    return rows, selected, counts


def public_features(rows):
    pos = np.asarray([row["pos"] for row in rows], dtype=np.float64)
    length = np.asarray([row["length"] for row in rows], dtype=np.float64)
    rel = pos / np.maximum(length - 1, 1)
    logits = np.vstack([row["logits"] for row in rows])
    return np.concatenate(
        [
            np.stack(
                [
                    pos,
                    rel,
                    length,
                    np.log(length),
                    (pos == 1).astype(np.float64),
                    (pos >= length - 2).astype(np.float64),
                ],
                axis=1,
            ),
            logits,
        ],
        axis=1,
    )


def leak_features(rows, rng):
    variances = np.vstack([row["variance"] for row in rows])
    k = rng.integers(1, LEAK_CONFIG["candidates"] + 1, size=variances.shape)
    return np.log(variances / LEAK_CONFIG["scale"] + LEAK_CONFIG["eps"]) + 2.0 * np.log(k)


def simulated_share_features(row_count, collection, share_dims, rng):
    full_dims = int(collection["module_count"]) * int(collection["hidden_size"])
    dims = full_dims if share_dims == 0 else min(share_dims, full_dims)
    # A single additive share is uniformly random and independent of the secret.
    # Standardized random features stress-test whether the classifier can extract
    # any label signal from share-shaped noise.
    return rng.standard_normal((row_count, dims)).astype(np.float32), dims, full_dims


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def topk_accuracy(scores, classes, y_true, k):
    k = min(k, scores.shape[1])
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.any(classes[top_idx] == y_true[:, None], axis=1).mean())


def prior_metrics(train_y, test_y):
    counts = Counter(train_y.tolist())
    ordered = [token_id for token_id, _ in counts.most_common()]
    out = {}
    for k in [1, 5, 10]:
        top = set(ordered[:k])
        out[f"prior_top{k}"] = float(np.asarray([v in top for v in test_y]).mean())
    return out


def nearest_centroid_metrics(X, y, groups, folds):
    n_splits = min(folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    metrics = {
        "top1": [],
        "top5": [],
        "top10": [],
        "prior_top1": [],
        "prior_top5": [],
        "prior_top10": [],
        "train_classes": [],
    }

    for train_idx, test_idx in cv.split(X, y, groups):
        train_classes = np.unique(y[train_idx])
        known = np.isin(y[test_idx], train_classes)
        test_idx = test_idx[known]
        if test_idx.size == 0:
            continue
        train_x, test_x = standardize(X[train_idx], X[test_idx])
        classes = train_classes
        centroids = np.vstack([train_x[y[train_idx] == cls].mean(axis=0) for cls in classes])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        test_norm = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
        scores = test_norm @ centroids.T
        pred = classes[np.argmax(scores, axis=1)]
        test_y = y[test_idx]
        metrics["top1"].append(float(accuracy_score(test_y, pred)))
        for k in [5, 10]:
            metrics[f"top{k}"].append(topk_accuracy(scores, classes, test_y, k))
        prior = prior_metrics(y[train_idx], test_y)
        for key, value in prior.items():
            metrics[key].append(value)
        metrics["train_classes"].append(len(classes))

    return {
        key: float(np.mean(values))
        for key, values in metrics.items()
        if values and key != "train_classes"
    } | {
        "folds": len(metrics["top1"]),
        "mean_train_classes": float(np.mean(metrics["train_classes"])),
    }


def write_markdown(path, report):
    lines = [
        "# LayerNorm Paper-Inspired Attack",
        "",
        "## Setup",
        "",
        f"- Samples: `{report['collection']['samples_used']}`",
        f"- Target token classes: `{report['target']['class_count']}`",
        f"- Covered token rows: `{report['target']['covered_rows']}` "
        f"({report['target']['coverage']:.2%})",
        f"- LayerNorm modules observed: `{report['collection']['module_count']}`",
        f"- Simulated share dims: `{report['share_view']['dims_used']}` "
        f"/ `{report['share_view']['full_dims']}`",
        f"- Classifier: nearest centroid with grouped sentence folds",
        "",
        "## Results",
        "",
        "| feature set | attacker interpretation | top1 | top5 | top10 | prior top1 | prior top5 | prior top10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['feature_set']} | {row['attacker_interpretation']} | "
            f"{m['top1']:.4f} | {m['top5']:.4f} | {m['top10']:.4f} | "
            f"{m['prior_top1']:.4f} | {m['prior_top5']:.4f} | {m['prior_top10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This stress test adapts the papers' token-cluster view to the values "
            "available in a semi-honest party's view: the actual revealed `log W` "
            "and a simulated fresh single-party activation share. A correctly "
            "generated additive share should be independent of token identity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    records, collection = collect_records(args, device)
    rows, selected, counts = select_rows(records, args.top_tokens, args.min_count)
    y = np.asarray([row["token_id"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["sample_id"] for row in rows], dtype=np.int64)

    logw = leak_features(rows, rng)
    share_x, share_dims_used, share_full_dims = simulated_share_features(
        len(rows), collection, args.share_dims, rng
    )
    feature_sets = [
        (
            "public",
            "no-leak baseline: position, length, final logits",
            public_features(rows),
        ),
        (
            "logW_protocol",
            "actual LayerNorm protocol leakage under frac_5M_S50",
            logw,
        ),
        (
            "share_only",
            "single-party secret share of intermediate activations",
            share_x,
        ),
        (
            "share+logW_protocol",
            "single-party activation share plus actual LayerNorm leakage",
            np.concatenate([share_x, logw.astype(np.float32)], axis=1),
        ),
        (
            "public+logW_protocol",
            "baseline plus actual LayerNorm protocol leakage",
            np.concatenate([public_features(rows), logw], axis=1),
        ),
    ]

    results = []
    for name, interpretation, X in feature_sets:
        results.append(
            {
                "feature_set": name,
                "attacker_interpretation": interpretation,
                "metrics": nearest_centroid_metrics(X, y, groups, args.folds),
            }
        )

    token_lookup = {row["token_id"]: row["token"] for row in rows}
    report = {
        "run_id": output_dir.name,
        "seed": args.seed,
        "folds": args.folds,
        "attacker": "nearest_centroid",
        "model_path": args.model_path,
        "validation_file": args.validation_file,
        "collection": collection,
        "share_view": {
            "dims_used": int(share_dims_used),
            "full_dims": int(share_full_dims),
            "simulation": "fresh additive share; independent of the secret",
        },
        "leak_config": LEAK_CONFIG,
        "target": {
            "top_tokens": args.top_tokens,
            "min_count": args.min_count,
            "class_count": len(selected),
            "covered_rows": len(rows),
            "coverage": len(rows) / max(1, len(records)),
            "token_vocab": [
                {
                    "token_id": int(token_id),
                    "token": token_lookup[token_id],
                    "count": int(counts[token_id]),
                }
                for token_id in selected
            ],
        },
        "results": results,
    }
    (output_dir / "paper_inspired_attack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "paper_inspired_attack.md", report)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/layernorm/layernorm_paper_inspired_attack.py",
                "argv": vars(args),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "target_rows": len(rows),
                "classes": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
