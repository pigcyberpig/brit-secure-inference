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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = os.path.join(os.environ.get("DATA_ROOT", ""), "bert-base-cased-sst2")
DEFAULT_VALIDATION = (
    "artifacts/experiment/sst2_scaled_tradeoff_20260524/"
    "half_subset_seed20260524/validation.parquet"
)


CONFIGS = [
    {
        "name": "int_z8_S1",
        "candidates": 255,
        "scale": 1.0,
        "eps": 1e-5,
    },
    {
        "name": "int_z14_S10",
        "candidates": 16_383,
        "scale": 10.0,
        "eps": 1e-5,
    },
    {
        "name": "frac_5M_S50",
        "candidates": 5_000_000,
        "scale": 50.0,
        "eps": 1e-5,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--validation_file", default=DEFAULT_VALIDATION)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=436)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260527)
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


def collect_sample_module_variances(args, device):
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
    module_to_idx = {name: idx for idx, name in enumerate(module_names)}
    sample_vars = [
        [None for _ in module_names]
        for _ in range(n_samples)
    ]
    current = {"sample_ids": None, "mask": None}
    hooks = []

    def make_hook(name):
        module_idx = module_to_idx[name]

        def hook(_module, inputs):
            x = inputs[0].detach().float()
            if x.dim() != 3:
                return
            var = x.var(dim=-1, unbiased=False).detach().cpu().numpy()
            mask = current["mask"]
            sample_ids = current["sample_ids"]
            if mask is None or sample_ids is None:
                return
            mask_np = mask.detach().cpu().numpy().astype(bool)
            for row_idx, sample_id in enumerate(sample_ids):
                sample_vars[sample_id][module_idx] = var[row_idx][mask_np[row_idx]].astype(
                    np.float64
                )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    labels = []
    lengths = []
    idxs = []
    start = time.time()
    try:
        with torch.no_grad():
            for offset in range(0, n_samples, args.batch_size):
                end = min(offset + args.batch_size, n_samples)
                batch_rows = dataset.select(range(offset, end))
                encoded = tokenizer(
                    batch_rows["sentence"],
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                current["sample_ids"] = list(range(offset, end))
                current["mask"] = encoded["attention_mask"].to(device)
                labels.extend(int(v) for v in batch_rows["label"])
                idxs.extend(int(v) for v in batch_rows["idx"])
                lengths.extend(int(v) for v in encoded["attention_mask"].sum(dim=1).tolist())
                model(**{k: v.to(device) for k, v in encoded.items()})
    finally:
        for hook in hooks:
            hook.remove()

    meta = {
        "samples_used": n_samples,
        "module_names": module_names,
        "module_count": len(module_names),
        "valid_token_count": int(sum(lengths)),
        "elapsed_s": time.time() - start,
    }
    return sample_vars, np.asarray(labels), np.asarray(lengths), np.asarray(idxs), meta


def stats_features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(9, dtype=np.float64)
    qs = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return np.asarray(
        [
            values.mean(),
            values.std(),
            *qs.tolist(),
        ],
        dtype=np.float64,
    )


def build_public_features(lengths):
    x = lengths.astype(np.float64)
    return np.stack(
        [
            x,
            np.log(x),
            (x <= np.quantile(x, 0.25)).astype(np.float64),
            (x >= np.quantile(x, 0.75)).astype(np.float64),
        ],
        axis=1,
    )


def simulate_logw_features(sample_vars, config, rng):
    rows = []
    for modules in sample_vars:
        all_logw = []
        per_module_stats = []
        for values in modules:
            if values is None:
                values = np.asarray([], dtype=np.float64)
            else:
                values = np.asarray(values, dtype=np.float64)
            if values.size:
                k = rng.integers(1, config["candidates"] + 1, size=values.size)
                logw = np.log(values / config["scale"] + config["eps"]) + 2.0 * np.log(k)
            else:
                logw = np.asarray([], dtype=np.float64)
            all_logw.append(logw)
            per_module_stats.append(stats_features(logw))
        flat = np.concatenate(all_logw) if all_logw else np.asarray([], dtype=np.float64)
        global_stats = stats_features(flat)
        module_stats = np.concatenate(per_module_stats)
        rows.append(np.concatenate([global_stats, module_stats]))
    return np.vstack(rows)


def length_bucket_targets(lengths):
    # Binary extremes are more stable than 4-way bins on a 436-sample split.
    q25 = np.quantile(lengths, 0.25)
    q75 = np.quantile(lengths, 0.75)
    short = (lengths <= q25).astype(np.int64)
    long = (lengths >= q75).astype(np.int64)
    return {
        "is_short_len_q25": short,
        "is_long_len_q75": long,
    }


def evaluate_binary(X, y, folds, seed, model_name):
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    min_class = int(np.bincount(y).min())
    n_splits = min(folds, min_class)
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        if model_name == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
        elif model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=seed,
            )
        else:
            raise ValueError(model_name)
        clf.fit(X[train_idx], y[train_idx])
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X[test_idx])[:, 1]
        else:
            scores = clf.decision_function(X[test_idx])
        pred = (scores >= 0.5).astype(np.int64)
        aucs.append(roc_auc_score(y[test_idx], scores))
        accs.append(accuracy_score(y[test_idx], pred))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "folds": n_splits,
        "positive_rate": float(y.mean()),
    }


def summarize_results(results):
    by_task = defaultdict(list)
    for row in results:
        by_task[row["task"]].append(row)
    summary = {}
    for task, rows in by_task.items():
        public = [r for r in rows if r["feature_set"] == "public"]
        best_public_auc = max((r["metrics"]["auc_mean"] for r in public), default=float("nan"))
        best_mask = max(
            (r for r in rows if r["feature_set"].startswith("public+")),
            key=lambda r: r["metrics"]["auc_mean"],
            default=None,
        )
        summary[task] = {
            "best_public_auc": best_public_auc,
            "best_mask_config": best_mask["feature_set"] if best_mask else None,
            "best_mask_auc": best_mask["metrics"]["auc_mean"] if best_mask else None,
            "best_auc_gain": (
                best_mask["metrics"]["auc_mean"] - best_public_auc if best_mask else None
            ),
        }
    return summary


def permutation_label_controls(feature_sets, labels, folds, seed):
    rng = np.random.default_rng(seed + 991)
    shuffled = np.asarray(labels, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    rows = []
    for feature_name, X in feature_sets.items():
        for model_name in ["logreg", "rf"]:
            metrics = evaluate_binary(X, shuffled, folds, seed + 991, model_name)
            if metrics is None:
                continue
            rows.append(
                {
                    "task": "label_permuted",
                    "feature_set": feature_name,
                    "model": model_name,
                    "metrics": metrics,
                }
            )
    return rows


def write_markdown(path, report):
    lines = [
        "# LayerNorm Mask Attribute Attack",
        "",
        "## Setup",
        "",
        f"- Samples: `{report['collection']['samples_used']}`",
        f"- LayerNorm modules: `{report['collection']['module_count']}`",
        f"- Valid tokens: `{report['collection']['valid_token_count']}`",
        f"- Folds: `{report['folds']}`",
        "",
        "## Best AUC Gain Over Public Baseline",
        "",
        "| task | public AUC | best masked feature | masked AUC | gain |",
        "|---|---:|---|---:|---:|",
    ]
    for task, row in report["summary"].items():
        lines.append(
            f"| {task} | {row['best_public_auc']:.4f} | {row['best_mask_config']} | "
            f"{row['best_mask_auc']:.4f} | {row['best_auc_gain']:.4f} |"
        )
    lines.extend(["", "## Full Results", ""])
    lines.append("| task | feature_set | model | AUC mean | AUC std | Acc mean | positive rate |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['task']} | {row['feature_set']} | {row['model']} | "
            f"{m['auc_mean']:.4f} | {m['auc_std']:.4f} | {m['acc_mean']:.4f} | "
            f"{m['positive_rate']:.4f} |"
        )
    lines.extend(["", "## Permuted Label Control", ""])
    lines.append("| feature_set | model | AUC mean | AUC std | Acc mean |")
    lines.append("|---|---|---:|---:|---:|")
    for row in report["permutation_results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['feature_set']} | {row['model']} | "
            f"{m['auc_mean']:.4f} | {m['auc_std']:.4f} | {m['acc_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    sample_vars, labels, lengths, idxs, collection = collect_sample_module_variances(
        args, device
    )
    public_X = build_public_features(lengths)
    targets = {"label": labels}
    targets.update(length_bucket_targets(lengths))

    feature_sets = {"public": public_X}
    for config in CONFIGS:
        mask_X = simulate_logw_features(sample_vars, config, rng)
        feature_sets[f"public+{config['name']}"] = np.concatenate([public_X, mask_X], axis=1)
        feature_sets[f"mask_only_{config['name']}"] = mask_X

    results = []
    for task, y in targets.items():
        for feature_name, X in feature_sets.items():
            for model_name in ["logreg", "rf"]:
                metrics = evaluate_binary(X, y, args.folds, args.seed, model_name)
                if metrics is None:
                    continue
                results.append(
                    {
                        "task": task,
                        "feature_set": feature_name,
                        "model": model_name,
                        "metrics": metrics,
                    }
                )

    permutation_results = permutation_label_controls(
        feature_sets, labels, args.folds, args.seed
    )

    report = {
        "run_id": output_dir.name,
        "seed": args.seed,
        "folds": args.folds,
        "model_path": args.model_path,
        "validation_file": args.validation_file,
        "collection": collection,
        "configs": CONFIGS,
        "results": results,
        "permutation_results": permutation_results,
        "summary": summarize_results(results),
    }
    (output_dir / "attribute_attack.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "attribute_attack.md", report)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/layernorm/layernorm_mask_attribute_attack.py",
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
    print(json.dumps({"output_dir": str(output_dir), "tasks": list(targets)}, indent=2))


if __name__ == "__main__":
    main()
