import argparse
import json
import os
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
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


LEAK_CONFIG = {
    "name": "frac_5M_S50",
    "candidates": 5_000_000,
    "scale": 50.0,
    "eps": 1e-5,
}


SENSITIVE_WORDS = {
    "bad", "best", "boring", "charm", "comedy", "dull", "enjoy", "excellent",
    "fails", "fun", "funny", "good", "great", "laugh", "like", "love", "moving",
    "poor", "powerful", "smart", "solid", "terrible", "touching", "worst",
    "worth",
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
    parser.add_argument("--top_tokens", type=int, default=100)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--attacker", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--mlp_epochs", type=int, default=80)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_patience", type=int, default=10)
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


def normalize_token(token):
    token = token.lower()
    token = token[2:] if token.startswith("##") else token
    token = re.sub(r"[^a-z]", "", token)
    return token


def is_sensitive_token(token):
    return normalize_token(token) in SENSITIVE_WORDS


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
    module_to_idx = {name: idx for idx, name in enumerate(module_names)}
    sample_vars = [[None for _ in module_names] for _ in range(n_samples)]
    sample_token_ids = [None for _ in range(n_samples)]
    sample_logits = np.zeros((n_samples, model.config.num_labels), dtype=np.float64)
    sample_lengths = np.zeros(n_samples, dtype=np.int64)
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
                    sample_token_ids[sample_id] = encoded["input_ids"][local_idx][
                        valid
                    ].cpu().numpy().astype(np.int64)
                    sample_lengths[sample_id] = int(valid.sum().item())
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
        stacked = np.stack(sample_vars[sample_id], axis=1)
        length = int(sample_lengths[sample_id])
        for pos, token_id in enumerate(token_ids.tolist()):
            if token_id in special_ids:
                continue
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            records.append(
                {
                    "sample_id": sample_id,
                    "pos": pos,
                    "length": length,
                    "token_id": int(token_id),
                    "token": token,
                    "sensitive": bool(is_sensitive_token(token)),
                    "variance": stacked[pos].astype(np.float64),
                    "logits": sample_logits[sample_id].astype(np.float64),
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


def select_target_vocab(records, top_tokens, min_count):
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


def oracle_features(rows):
    variances = np.vstack([row["variance"] for row in rows])
    return np.log(variances + 1e-5)


def topk_accuracy(proba, classes, y_true, k, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    if not np.any(mask):
        return None
    proba = proba[mask]
    y_true = y_true[mask]
    k = min(k, proba.shape[1])
    top_idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    top_classes = classes[top_idx]
    return float(np.any(top_classes == y_true[:, None], axis=1).mean())


def prior_metrics(train_y, test_y, sensitive_mask):
    counts = Counter(train_y.tolist())
    ordered = [token_id for token_id, _ in counts.most_common()]
    out = {}
    for k in [1, 5, 10]:
        top = set(ordered[:k])
        out[f"top{k}"] = float(np.asarray([v in top for v in test_y]).mean())
        if np.any(sensitive_mask):
            out[f"sensitive_recall@{k}"] = float(
                np.asarray([v in top for v in test_y[sensitive_mask]]).mean()
            )
        else:
            out[f"sensitive_recall@{k}"] = None
    return out


class MLPAttacker(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def mlp_predict_proba(train_x, train_y, test_x, seed, args):
    train_x, test_x = standardize(train_x.astype(np.float32), test_x.astype(np.float32))
    classes = np.unique(train_y)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    y_idx = np.asarray([class_to_idx[int(v)] for v in train_y], dtype=np.int64)

    rng = np.random.default_rng(seed)
    val_size = max(len(train_x) // 10, len(classes))
    val_size = min(val_size, len(train_x) // 3)
    perm = rng.permutation(len(train_x))
    val_idx = perm[:val_size]
    fit_idx = perm[val_size:]
    if fit_idx.size == 0:
        fit_idx = perm
        val_idx = perm[:0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = MLPAttacker(train_x.shape[1], args.mlp_hidden, len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=1e-4)
    counts = np.bincount(y_idx, minlength=len(classes)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    x_fit = torch.tensor(train_x[fit_idx], dtype=torch.float32)
    y_fit = torch.tensor(y_idx[fit_idx], dtype=torch.long)
    x_val = torch.tensor(train_x[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_idx[val_idx], dtype=torch.long, device=device)
    best_state = None
    best_val = float("inf")
    stale = 0
    batch_size = min(args.mlp_batch_size, len(x_fit))

    for _epoch in range(args.mlp_epochs):
        model.train()
        order = torch.randperm(len(x_fit))
        for start in range(0, len(x_fit), batch_size):
            idx = order[start : start + batch_size]
            xb = x_fit[idx].to(device)
            yb = y_fit[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if len(val_idx):
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_val), y_val).item())
        else:
            val_loss = 0.0
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.mlp_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_x, dtype=torch.float32, device=device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    return classes, proba


def logreg_predict_proba(train_x, train_y, test_x, seed):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    clf.fit(train_x, train_y)
    return clf.classes_, clf.predict_proba(test_x)


def evaluate_setting(X, y, groups, sensitive_mask, folds, seed, args):
    n_splits = min(folds, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    metrics = {
        "top1": [],
        "top5": [],
        "top10": [],
        "sensitive_recall@1": [],
        "sensitive_recall@5": [],
        "sensitive_recall@10": [],
        "prior_top1": [],
        "prior_top5": [],
        "prior_top10": [],
        "prior_sensitive_recall@1": [],
        "prior_sensitive_recall@5": [],
        "prior_sensitive_recall@10": [],
    }
    for train_idx, test_idx in cv.split(X, y, groups):
        train_classes = np.unique(y[train_idx])
        known = np.isin(y[test_idx], train_classes)
        test_idx = test_idx[known]
        if test_idx.size == 0:
            continue
        if args.attacker == "logreg":
            classes, proba = logreg_predict_proba(
                X[train_idx], y[train_idx], X[test_idx], seed
            )
        else:
            classes, proba = mlp_predict_proba(
                X[train_idx], y[train_idx], X[test_idx], seed, args
            )
        pred = classes[np.argmax(proba, axis=1)]
        test_y = y[test_idx]
        test_sensitive = sensitive_mask[test_idx]
        metrics["top1"].append(float(accuracy_score(test_y, pred)))
        for k in [5, 10]:
            metrics[f"top{k}"].append(topk_accuracy(proba, classes, test_y, k))
        for k in [1, 5, 10]:
            v = topk_accuracy(proba, classes, test_y, k, test_sensitive)
            if v is not None:
                metrics[f"sensitive_recall@{k}"].append(v)
        prior = prior_metrics(y[train_idx], test_y, test_sensitive)
        for k in [1, 5, 10]:
            metrics[f"prior_top{k}"].append(prior[f"top{k}"])
            if prior[f"sensitive_recall@{k}"] is not None:
                metrics[f"prior_sensitive_recall@{k}"].append(prior[f"sensitive_recall@{k}"])
    return {key: float(np.mean(vals)) for key, vals in metrics.items() if vals}


def write_markdown(path, report):
    lines = [
        "# LayerNorm Token Recovery: Three Settings",
        "",
        "## Setup",
        "",
        f"- Samples: `{report['collection']['samples_used']}`",
        f"- Target token classes: `{report['target']['class_count']}`",
        f"- Covered token rows: `{report['target']['covered_rows']}` "
        f"({report['target']['coverage']:.2%})",
        f"- Sensitive token rows: `{report['target']['sensitive_rows']}`",
        f"- Leak config: `{LEAK_CONFIG['name']}`",
        f"- Attacker: `{report['attacker']}`",
        "",
        "## Results",
        "",
        "| setting | attacker sees | top1 | top5 | top10 | sensitive R@1 | sensitive R@5 | sensitive R@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['setting']} | {row['attacker_sees']} | "
            f"{m['top1']:.4f} | {m['top5']:.4f} | {m['top10']:.4f} | "
            f"{m.get('sensitive_recall@1', float('nan')):.4f} | "
            f"{m.get('sensitive_recall@5', float('nan')):.4f} | "
            f"{m.get('sensitive_recall@10', float('nan')):.4f} |"
        )
    lines.extend(["", "## Frequency Prior", ""])
    lines.append("| setting | prior top1 | prior top5 | prior top10 | prior sensitive R@5 | prior sensitive R@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['setting']} | {m['prior_top1']:.4f} | {m['prior_top5']:.4f} | "
            f"{m['prior_top10']:.4f} | {m.get('prior_sensitive_recall@5', float('nan')):.4f} | "
            f"{m.get('prior_sensitive_recall@10', float('nan')):.4f} |"
        )
    lines.extend(["", "## Sensitive Tokens In Target Vocab", ""])
    lines.append(", ".join(report["target"]["sensitive_tokens"]) or "None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    records, collection = collect_records(args, device)
    selected, counts = select_target_vocab(records, args.top_tokens, args.min_count)
    selected_set = set(selected)
    rows = [row for row in records if row["token_id"] in selected_set]
    y = np.asarray([row["token_id"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["sample_id"] for row in rows], dtype=np.int64)
    sensitive_mask = np.asarray([row["sensitive"] for row in rows], dtype=bool)

    public_x = public_features(rows)
    leak_x = leak_features(rows, rng)
    oracle_x = oracle_features(rows)
    feature_sets = [
        ("No-leak baseline", "position / length / final logits", public_x),
        ("Leak", "No-leak + log W", np.concatenate([public_x, leak_x], axis=1)),
        ("Oracle", "No-leak + true log X", np.concatenate([public_x, oracle_x], axis=1)),
    ]

    results = []
    for setting, sees, X in feature_sets:
        results.append(
            {
                "setting": setting,
                "attacker_sees": sees,
                "metrics": evaluate_setting(
                    X, y, groups, sensitive_mask, args.folds, args.seed, args
                ),
            }
        )

    token_lookup = {row["token_id"]: row["token"] for row in rows}
    sensitive_tokens = sorted(
        {
            token_lookup[token_id]
            for token_id in selected
            if any(row["token_id"] == token_id and row["sensitive"] for row in rows)
        }
    )
    report = {
        "run_id": output_dir.name,
        "seed": args.seed,
        "folds": args.folds,
        "attacker": args.attacker,
        "model_path": args.model_path,
        "validation_file": args.validation_file,
        "collection": collection,
        "leak_config": LEAK_CONFIG,
        "target": {
            "top_tokens": args.top_tokens,
            "min_count": args.min_count,
            "class_count": len(selected),
            "covered_rows": len(rows),
            "coverage": len(rows) / max(1, len(records)),
            "sensitive_rows": int(sensitive_mask.sum()),
            "sensitive_tokens": sensitive_tokens,
            "token_vocab": [
                {
                    "token_id": int(token_id),
                    "token": token_lookup[token_id],
                    "count": int(counts[token_id]),
                    "sensitive": token_lookup[token_id] in sensitive_tokens,
                }
                for token_id in selected
            ],
        },
        "results": results,
    }
    (output_dir / "token_recovery_three_settings.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "token_recovery_three_settings.md", report)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/layernorm/layernorm_token_recovery_three_settings.py",
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
                "sensitive_rows": int(sensitive_mask.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
