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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
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
    {"name": "int_z8_S1", "candidates": 255, "scale": 1.0, "eps": 1e-5},
    {"name": "int_z14_S10", "candidates": 16_383, "scale": 10.0, "eps": 1e-5},
    {"name": "frac_5M_S50", "candidates": 5_000_000, "scale": 50.0, "eps": 1e-5},
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
    parser.add_argument("--top_tokens", type=int, default=50)
    parser.add_argument("--min_count", type=int, default=5)
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


def collect_token_records(args, device):
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
    sample_vars = [[None for _ in module_names] for _ in range(n_samples)]
    sample_token_ids = [None for _ in range(n_samples)]
    sample_labels = np.zeros(n_samples, dtype=np.int64)
    sample_lengths = np.zeros(n_samples, dtype=np.int64)
    sample_dataset_idx = np.zeros(n_samples, dtype=np.int64)
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
                mask = encoded["attention_mask"]
                current["sample_ids"] = list(range(offset, end))
                current["mask"] = mask.to(device)
                for local_idx, sample_id in enumerate(range(offset, end)):
                    valid = mask[local_idx].bool()
                    sample_token_ids[sample_id] = encoded["input_ids"][local_idx][
                        valid
                    ].cpu().numpy().astype(np.int64)
                    sample_labels[sample_id] = int(batch_rows["label"][local_idx])
                    sample_dataset_idx[sample_id] = int(batch_rows["idx"][local_idx])
                    sample_lengths[sample_id] = int(valid.sum().item())
                model(**{k: v.to(device) for k, v in encoded.items()})
    finally:
        for hook in hooks:
            hook.remove()

    special_ids = set(tokenizer.all_special_ids)
    records = []
    for sample_id in range(n_samples):
        token_ids = sample_token_ids[sample_id]
        length = int(sample_lengths[sample_id])
        if token_ids is None:
            continue
        stacked = np.stack(sample_vars[sample_id], axis=1)
        for pos, token_id in enumerate(token_ids.tolist()):
            if token_id in special_ids:
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "dataset_idx": int(sample_dataset_idx[sample_id]),
                    "pos": pos,
                    "length": length,
                    "label": int(sample_labels[sample_id]),
                    "token_id": int(token_id),
                    "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                    "variance": stacked[pos].astype(np.float64),
                }
            )

    meta = {
        "samples_used": n_samples,
        "module_count": len(module_names),
        "module_names": module_names,
        "content_token_count": len(records),
        "elapsed_s": time.time() - start,
    }
    return records, meta


def choose_target_tokens(records, top_tokens, min_count):
    counts = Counter(row["token_id"] for row in records)
    selected = [
        token_id
        for token_id, count in counts.most_common()
        if count >= min_count
    ][:top_tokens]
    return selected, counts


def public_features(rows):
    pos = np.asarray([row["pos"] for row in rows], dtype=np.float64)
    length = np.asarray([row["length"] for row in rows], dtype=np.float64)
    rel = pos / np.maximum(length - 1, 1)
    return np.stack(
        [
            pos,
            rel,
            length,
            np.log(length),
            (pos == 1).astype(np.float64),
            (pos >= length - 2).astype(np.float64),
        ],
        axis=1,
    )


def mask_features(rows, config, rng):
    variances = np.vstack([row["variance"] for row in rows])
    k = rng.integers(1, config["candidates"] + 1, size=variances.shape)
    return np.log(variances / config["scale"] + config["eps"]) + 2.0 * np.log(k)


def topk_accuracy(proba, classes, y_true, k):
    k = min(k, proba.shape[1])
    top_idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    top_classes = classes[top_idx]
    return float(np.any(top_classes == y_true[:, None], axis=1).mean())


def prior_metrics(y_true, train_y):
    counts = Counter(train_y.tolist())
    ordered = [token_id for token_id, _ in counts.most_common()]
    top1 = ordered[0]
    top5 = set(ordered[:5])
    return {
        "top1_acc": float((y_true == top1).mean()),
        "top5_acc": float(np.asarray([v in top5 for v in y_true]).mean()),
    }


def evaluate_multiclass(X, y, groups, folds, seed, model_name):
    n_splits = min(folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    top1 = []
    top5 = []
    prior_top1 = []
    prior_top5 = []
    class_counts = []
    for train_idx, test_idx in cv.split(X, y, groups):
        train_classes = np.unique(y[train_idx])
        keep_test = np.isin(y[test_idx], train_classes)
        test_idx = test_idx[keep_test]
        if test_idx.size == 0:
            continue
        if model_name == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                    multi_class="auto",
                ),
            )
        elif model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
            )
        else:
            raise ValueError(model_name)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])
        pred = clf.classes_[np.argmax(proba, axis=1)]
        top1.append(accuracy_score(y[test_idx], pred))
        top5.append(topk_accuracy(proba, clf.classes_, y[test_idx], 5))
        prior = prior_metrics(y[test_idx], y[train_idx])
        prior_top1.append(prior["top1_acc"])
        prior_top5.append(prior["top5_acc"])
        class_counts.append(len(clf.classes_))
    return {
        "top1_acc_mean": float(np.mean(top1)),
        "top1_acc_std": float(np.std(top1)),
        "top5_acc_mean": float(np.mean(top5)),
        "top5_acc_std": float(np.std(top5)),
        "prior_top1_acc_mean": float(np.mean(prior_top1)),
        "prior_top5_acc_mean": float(np.mean(prior_top5)),
        "folds": len(top1),
        "mean_train_classes": float(np.mean(class_counts)),
    }


def write_markdown(path, report):
    lines = [
        "# LayerNorm Token Reconstruction Attack",
        "",
        "## Setup",
        "",
        f"- Samples: `{report['collection']['samples_used']}`",
        f"- Content tokens: `{report['collection']['content_token_count']}`",
        f"- LayerNorm modules: `{report['collection']['module_count']}`",
        f"- Target token classes: `{report['target']['class_count']}`",
        f"- Covered token rows: `{report['target']['covered_rows']}` "
        f"({report['target']['coverage']:.2%})",
        "",
        "## Results",
        "",
        "| feature_set | model | top1 | top5 | prior top1 | prior top5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['feature_set']} | {row['model']} | "
            f"{m['top1_acc_mean']:.4f} | {m['top5_acc_mean']:.4f} | "
            f"{m['prior_top1_acc_mean']:.4f} | {m['prior_top5_acc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The attack predicts frequent non-special token identity from one revealed masked "
            "LayerNorm variance vector per token. Evaluation is grouped by sentence, so no "
            "sentence contributes tokens to both train and test folds.",
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

    records, collection = collect_token_records(args, device)
    selected, counts = choose_target_tokens(records, args.top_tokens, args.min_count)
    selected_set = set(selected)
    rows = [row for row in records if row["token_id"] in selected_set]
    y = np.asarray([row["token_id"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["sample_id"] for row in rows], dtype=np.int64)

    feature_sets = {"public": public_features(rows)}
    for config in CONFIGS:
        mask_X = mask_features(rows, config, rng)
        public_X = feature_sets["public"]
        feature_sets[f"mask_only_{config['name']}"] = mask_X
        feature_sets[f"public+{config['name']}"] = np.concatenate([public_X, mask_X], axis=1)

    results = []
    for feature_name, X in feature_sets.items():
        for model_name in ["logreg", "rf"]:
            metrics = evaluate_multiclass(X, y, groups, args.folds, args.seed, model_name)
            results.append(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "metrics": metrics,
                }
            )

    token_vocab = [
        {
            "token_id": int(token_id),
            "token": next(row["token"] for row in records if row["token_id"] == token_id),
            "count": int(counts[token_id]),
        }
        for token_id in selected
    ]
    report = {
        "run_id": output_dir.name,
        "seed": args.seed,
        "folds": args.folds,
        "model_path": args.model_path,
        "validation_file": args.validation_file,
        "collection": collection,
        "target": {
            "top_tokens": args.top_tokens,
            "min_count": args.min_count,
            "class_count": len(selected),
            "covered_rows": len(rows),
            "coverage": len(rows) / max(1, len(records)),
            "token_vocab": token_vocab,
        },
        "configs": CONFIGS,
        "results": results,
    }
    (output_dir / "token_reconstruction.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "token_reconstruction.md", report)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/layernorm/layernorm_token_reconstruction_attack.py",
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
                "content_tokens": len(records),
                "target_rows": len(rows),
                "classes": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
