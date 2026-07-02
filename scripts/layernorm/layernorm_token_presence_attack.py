import argparse
import json
import os
import platform
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    {"name": "int_z8_S1", "candidates": 255, "scale": 1.0, "eps": 1e-5},
    {"name": "int_z14_S10", "candidates": 16_383, "scale": 10.0, "eps": 1e-5},
    {"name": "frac_5M_S50", "candidates": 5_000_000, "scale": 50.0, "eps": 1e-5},
]

STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "you", "are", "was", "were",
    "have", "has", "had", "not", "but", "his", "her", "its", "they", "them",
    "from", "into", "about", "than", "then", "there", "their", "what", "when",
    "where", "who", "why", "how", "all", "one", "out", "just", "can", "will",
    "more", "most", "some", "such", "very", "too", "off", "our", "your",
    "it's", "isn", "don", "doesn", "didn", "film", "movie",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--validation_file", default=DEFAULT_VALIDATION)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--samples", type=int, default=436)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top_tokens", type=int, default=30)
    parser.add_argument("--min_doc_count", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["logreg", "rf"])
    parser.add_argument("--rf_estimators", type=int, default=300)
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


def token_is_content(token):
    if token.startswith("##"):
        return False
    if not re.search(r"[A-Za-z]", token):
        return False
    token_l = token.lower()
    if len(token_l) < 3:
        return False
    return token_l not in STOPWORDS


def collect_sample_data(args, device):
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
    sample_tokens = [None for _ in range(n_samples)]
    sample_lengths = np.zeros(n_samples, dtype=np.int64)
    sample_labels = np.zeros(n_samples, dtype=np.int64)
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
                    ids = encoded["input_ids"][local_idx][valid].cpu().numpy().astype(np.int64)
                    sample_token_ids[sample_id] = ids
                    sample_tokens[sample_id] = tokenizer.convert_ids_to_tokens(ids.tolist())
                    sample_lengths[sample_id] = int(valid.sum().item())
                    sample_labels[sample_id] = int(rows["label"][local_idx])
                model(**{k: v.to(device) for k, v in encoded.items()})
    finally:
        for hook in hooks:
            hook.remove()

    meta = {
        "samples_used": n_samples,
        "module_count": len(module_names),
        "module_names": module_names,
        "valid_token_count": int(sample_lengths.sum()),
        "elapsed_s": time.time() - start,
    }
    return sample_vars, sample_tokens, sample_lengths, sample_labels, meta


def stats_features(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(9, dtype=np.float64)
    qs = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return np.asarray([values.mean(), values.std(), *qs.tolist()], dtype=np.float64)


def sample_mask_features(sample_vars, config, rng):
    rows = []
    for modules in sample_vars:
        stacked = np.stack(modules, axis=1)
        k = rng.integers(1, config["candidates"] + 1, size=stacked.shape)
        logw = np.log(stacked / config["scale"] + config["eps"]) + 2.0 * np.log(k)
        global_stats = stats_features(logw.reshape(-1))
        layer_stats = np.concatenate([stats_features(logw[:, i]) for i in range(logw.shape[1])])
        rows.append(np.concatenate([global_stats, layer_stats]))
    return np.vstack(rows)


def public_features(lengths):
    x = lengths.astype(np.float64)
    return np.stack([x, np.log(x)], axis=1)


def select_tokens(sample_tokens, top_tokens, min_doc_count):
    doc_counts = Counter()
    for tokens in sample_tokens:
        present = {tok.lower() for tok in tokens if token_is_content(tok)}
        doc_counts.update(present)
    selected = [
        tok for tok, count in doc_counts.most_common(top_tokens) if count >= min_doc_count
    ]
    return selected, doc_counts


def evaluate_binary(X, y, folds, seed, model_name, rf_estimators):
    if len(np.unique(y)) < 2:
        return None
    min_class = int(np.bincount(y).min())
    n_splits = min(folds, min_class)
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        if model_name == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
        elif model_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=rf_estimators,
                max_depth=4,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=seed,
            )
        else:
            raise ValueError(model_name)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], scores))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def write_markdown(path, report):
    lines = [
        "# LayerNorm Token Presence Attack",
        "",
        "## Setup",
        "",
        f"- Samples: `{report['collection']['samples_used']}`",
        f"- Candidate content tokens: `{len(report['tokens'])}`",
        f"- LayerNorm modules: `{report['collection']['module_count']}`",
        "",
        "## Summary",
        "",
        "| feature_set | model | mean AUC | median AUC | tokens AUC>=0.7 | best token | best AUC |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['mean_auc']:.4f} | "
            f"{row['median_auc']:.4f} | {row['tokens_auc_ge_0_7']} | "
            f"{row['best_token']} | {row['best_auc']:.4f} |"
        )
    lines.extend(["", "## Per Token Best Mask-Only AUC", ""])
    lines.append("| token | doc_count | best mask-only AUC | best feature |")
    lines.append("|---|---:|---:|---|")
    for row in report["per_token_best"]:
        lines.append(
            f"| {row['token']} | {row['doc_count']} | {row['best_auc']:.4f} | {row['feature_set']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    sample_vars, sample_tokens, lengths, labels, collection = collect_sample_data(args, device)
    selected, doc_counts = select_tokens(sample_tokens, args.top_tokens, args.min_doc_count)
    feature_sets = {"public": public_features(lengths)}
    for config in CONFIGS:
        mask_X = sample_mask_features(sample_vars, config, rng)
        feature_sets[f"mask_only_{config['name']}"] = mask_X
        feature_sets[f"public+{config['name']}"] = np.concatenate(
            [feature_sets["public"], mask_X], axis=1
        )

    rows = []
    for token in selected:
        y = np.asarray([token in {t.lower() for t in toks} for toks in sample_tokens], dtype=np.int64)
        for feature_name, X in feature_sets.items():
            for model_name in args.models:
                metrics = evaluate_binary(
                    X, y, args.folds, args.seed, model_name, args.rf_estimators
                )
                if metrics is None:
                    continue
                rows.append(
                    {
                        "token": token,
                        "doc_count": int(doc_counts[token]),
                        "feature_set": feature_name,
                        "model": model_name,
                        "metrics": metrics,
                    }
                )

    summary = []
    for feature_name in feature_sets:
        for model_name in ["logreg", "rf"]:
            vals = [
                row for row in rows if row["feature_set"] == feature_name and row["model"] == model_name
            ]
            if not vals:
                continue
            aucs = np.asarray([row["metrics"]["auc_mean"] for row in vals])
            best = vals[int(np.argmax(aucs))]
            summary.append(
                {
                    "feature_set": feature_name,
                    "model": model_name,
                    "mean_auc": float(aucs.mean()),
                    "median_auc": float(np.median(aucs)),
                    "tokens_auc_ge_0_7": int((aucs >= 0.7).sum()),
                    "best_token": best["token"],
                    "best_auc": best["metrics"]["auc_mean"],
                }
            )

    per_token_best = []
    for token in selected:
        vals = [
            row
            for row in rows
            if row["token"] == token and row["feature_set"].startswith("mask_only")
        ]
        if not vals:
            continue
        best = max(vals, key=lambda row: row["metrics"]["auc_mean"])
        per_token_best.append(
            {
                "token": token,
                "doc_count": int(doc_counts[token]),
                "best_auc": best["metrics"]["auc_mean"],
                "feature_set": best["feature_set"],
                "model": best["model"],
            }
        )
    per_token_best.sort(key=lambda row: row["best_auc"], reverse=True)

    report = {
        "run_id": output_dir.name,
        "seed": args.seed,
        "folds": args.folds,
        "models": args.models,
        "collection": collection,
        "tokens": [{"token": t, "doc_count": int(doc_counts[t])} for t in selected],
        "configs": CONFIGS,
        "summary": summary,
        "per_token_best": per_token_best,
        "rows": rows,
    }
    (output_dir / "token_presence.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "token_presence.md", report)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/layernorm/layernorm_token_presence_attack.py",
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
    print(json.dumps({"output_dir": str(output_dir), "tokens": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
