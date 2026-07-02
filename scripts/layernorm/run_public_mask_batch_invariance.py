import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
TEXT_CLASSIFICATION_ROOT = str(get_data_root())

import crypten as ct
import datasets
import torch
import transformers
from crypten.config import cfg
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding


MODEL_PATH = f"{TEXT_CLASSIFICATION_ROOT}/bert-base-cased-sst2"
VALIDATION_FILE = f"{TEXT_CLASSIFICATION_ROOT}/glue/sst2/validation.parquet"
MAX_LENGTH = 128


def softmax_override(name):
    if name == "ode_clip_i16":
        return {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": True,
            "functions.softmax_ode_iter_num": 16,
        }
    if name.startswith("scaled_k") and "_i" in name:
        return {"functions.softmax_method": name}
    raise ValueError(f"unknown softmax config: {name}")


def parse_scaled_method(name):
    scale_part, iter_part = name[len("scaled_k") :].split("_i", 1)
    return int(scale_part), int(iter_part)


def mask_for_input(mask, input_tensor):
    public = mask.to(device=input_tensor.device, dtype=torch.float32)
    while public.dim() < input_tensor.dim():
        public = public.unsqueeze(1)
    return public


def masked_sum(x, mask, dim, keepdim=True):
    return (x * mask).sum(dim=dim, keepdim=keepdim)


def masked_mean(x, mask, dim, keepdim=True):
    total = masked_sum(x, mask, dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return total / count


def masked_ode_softmax(x, mask, dim, iter_num, clip, lower, upper):
    if clip:
        diff = ct.cat([x - upper, lower - x]).relu().split(x.shape[0])
        x = x + diff[1] - diff[0]

    x = x / iter_num
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    init = mask.expand(tuple(x.shape)) / count
    g = x.new(init, device=x.device)

    for _ in range(iter_num):
        gx_sum = masked_sum(g * x, mask, dim=dim, keepdim=True)
        g = g + ((x - gx_sum) * g)
        g = g * mask
    return g


def masked_scaled_softmax(x, mask, dim, method):
    scale, iter_num = parse_scaled_method(method)
    centered = x - masked_mean(x, mask, dim=dim, keepdim=True)
    scaled = centered / scale
    probs = masked_ode_softmax(
        scaled,
        mask,
        dim=dim,
        iter_num=iter_num,
        clip=False,
        lower=cfg.functions.softmax_ode_lb,
        upper=cfg.functions.softmax_ode_ub,
    )

    powered = probs
    if scale > 1:
        exponent = scale
        result = None
        base = probs
        while exponent:
            if exponent & 1:
                result = base if result is None else result * base
            exponent >>= 1
            if exponent:
                base = base * base
        powered = result

    denom = masked_sum(powered, mask, dim=dim, keepdim=True)
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_total = denom.reciprocal()
    return powered * inv_total * mask


def masked_softmax(x, mask, dim):
    method = cfg.functions.softmax_method
    public_mask = mask_for_input(mask, x)
    if method == "ode":
        return masked_ode_softmax(
            x,
            public_mask,
            dim=dim,
            iter_num=cfg.functions.softmax_ode_iter_num,
            clip=cfg.functions.softmax_ode_clip,
            lower=cfg.functions.softmax_ode_lb,
            upper=cfg.functions.softmax_ode_ub,
        )
    if method.startswith("scaled_k") and "_i" in method:
        return masked_scaled_softmax(x, public_mask, dim=dim, method=method)
    if method in {"ideal", "reciprocal"}:
        very_negative = (1.0 - public_mask) * -10000.0
        return (x + very_negative).softmax(dim)
    raise ValueError(f"masked softmax does not support method {method}")


class PublicMaskSoftmaxPatch:
    def __init__(self, private_model):
        self.private_model = private_model
        self.current_mask = None
        self.original_forwards = {}

    def set_mask(self, attention_mask):
        self.current_mask = attention_mask.detach()

    def install(self):
        for name, module in self.private_model._modules.items():
            if "/attention/self/Softmax_output_0" not in name:
                continue
            original = object.__getattribute__(module, "forward")
            self.original_forwards[name] = original

            def wrapped(module_self, input_tensor, _orig=original):
                if self.current_mask is None:
                    return _orig(input_tensor)
                return masked_softmax(input_tensor, self.current_mask, module_self.dim)

            module.forward = MethodType(wrapped, module)

    def uninstall(self):
        for name, original in self.original_forwards.items():
            self.private_model._modules[name].forward = original
        self.original_forwards.clear()


def load_sst2(max_samples):
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_error()
    raw = load_dataset("parquet", data_files={"validation": VALIDATION_FILE})["validation"]

    config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    model.eval()

    def preprocess(examples):
        result = tokenizer(
            examples["sentence"],
            padding=False,
            max_length=MAX_LENGTH,
            truncation=True,
        )
        result["labels"] = examples["label"]
        return result

    processed = raw.map(preprocess, batched=True, remove_columns=raw.column_names)
    if max_samples is not None:
        processed = Subset(processed, list(range(max_samples)))
    return model, tokenizer, processed


def reset_private_stats(private_model):
    for attr in (
        "embedding_time",
        "matmul_time",
        "softmax_time",
        "gelu_time",
        "layernorm_time",
        "tanh_time",
        "conv_time",
        "other_time",
    ):
        if hasattr(private_model, attr):
            setattr(private_model, attr, 0.0)


def run_private_eval(model, tokenizer, dataset, batch_size, softmax_config, sqrt_method, masked):
    data_loader = DataLoader(
        dataset,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=batch_size,
        shuffle=False,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    softmax_cfg = softmax_override(softmax_config)
    start = time.time()
    rows = []

    with cfg.temp_override({"functions.sqrt_method": sqrt_method}):
        with cfg.temp_override(softmax_cfg):
            dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
            private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
            patch = PublicMaskSoftmaxPatch(private_model)
            if masked:
                patch.install()

            sample_offset = 0
            for batch in data_loader:
                reset_private_stats(private_model)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids))
                labels = batch["labels"]
                if masked:
                    patch.set_mask(attention_mask)

                with ct.no_grad():
                    logits = private_model(
                        ct.cryptensor(input_ids).to(device),
                        ct.cryptensor(attention_mask).to(device),
                        ct.cryptensor(token_type_ids).to(device),
                    ).get_plain_text().cpu()

                preds = logits.argmax(dim=-1)
                lengths = attention_mask.sum(dim=-1)
                for pos in range(logits.size(0)):
                    rows.append(
                        {
                            "sample": sample_offset + pos,
                            "pred": int(preds[pos].item()),
                            "label": int(labels[pos].item()),
                            "correct": bool(preds[pos].item() == labels[pos].item()),
                            "length": int(lengths[pos].item()),
                            "padded_length": int(input_ids.size(1)),
                            "logits": [float(v) for v in logits[pos].tolist()],
                        }
                    )
                sample_offset += logits.size(0)

            if masked:
                patch.uninstall()

    correct = sum(1 for row in rows if row["correct"])
    return {
        "batch_size": batch_size,
        "masked": masked,
        "softmax_config": softmax_config,
        "sqrt_method": sqrt_method,
        "samples": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "runtime_s": time.time() - start,
        "rows": rows,
    }


@contextmanager
def crypten_ready():
    ct.init()
    try:
        yield
    finally:
        pass


def run_synthetic(softmax_config, sqrt_method):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0, -10000.0, -10000.0]],
            [[1.5, 0.5, -10000.0, -10000.0, -10000.0]],
        ],
        device=device,
    )
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
    mask3 = mask[:, None, :]

    with cfg.temp_override({"functions.sqrt_method": sqrt_method}):
        with cfg.temp_override(softmax_override(softmax_config)):
            additive = ct.cryptensor(logits).softmax(-1).get_plain_text()
            masked = masked_softmax(ct.cryptensor(logits), mask3, -1).get_plain_text()

    additive_pad_mass = ((additive * (1.0 - mask3.to(device))).sum(dim=-1)).cpu()
    masked_pad_mass = ((masked * (1.0 - mask3.to(device))).sum(dim=-1)).cpu()
    masked_row_sum = masked.sum(dim=-1).cpu()
    return {
        "softmax_config": softmax_config,
        "sqrt_method": sqrt_method,
        "additive_pad_mass_max": float(additive_pad_mass.max().item()),
        "masked_pad_mass_max": float(masked_pad_mass.max().item()),
        "masked_row_sum_max_abs_err": float((masked_row_sum - 1.0).abs().max().item()),
        "additive": additive.cpu().tolist(),
        "masked": masked.cpu().tolist(),
    }


def compare_runs(single, batched):
    by_sample = {row["sample"]: row for row in single["rows"]}
    flips = []
    max_logit_linf = 0.0
    for row in batched["rows"]:
        base = by_sample[row["sample"]]
        if row["pred"] != base["pred"]:
            flips.append(
                {
                    "sample": row["sample"],
                    "single_pred": base["pred"],
                    "batch_pred": row["pred"],
                    "label": row["label"],
                }
            )
        drift = max(abs(a - b) for a, b in zip(base["logits"], row["logits"]))
        max_logit_linf = max(max_logit_linf, drift)
    return {
        "accuracy_delta": batched["accuracy"] - single["accuracy"],
        "prediction_flip_count": len(flips),
        "prediction_flips": flips,
        "max_logit_linf": max_logit_linf,
    }


def write_summary(output_dir, payload):
    comparison = payload["comparison"]
    accepted = payload["accepted"]
    lines = [
        "# Public-Mask-Aware Batch Invariance Result",
        "",
        f"Accepted: `{accepted}`",
        "",
        "## Synthetic Operator Check",
        "",
        f"- additive pad mass max: `{payload['synthetic']['additive_pad_mass_max']:.8f}`",
        f"- masked pad mass max: `{payload['synthetic']['masked_pad_mass_max']:.8f}`",
        f"- masked row-sum max abs error: `{payload['synthetic']['masked_row_sum_max_abs_err']:.8f}`",
        "",
        "## SST-2 End-to-End Check",
        "",
        f"- samples: `{payload['single']['samples']}`",
        f"- softmax: `{payload['single']['softmax_config']}`",
        f"- sqrt method: `{payload['single']['sqrt_method']}`",
        f"- bs=1 accuracy: `{payload['single']['accuracy']:.8f}`",
        f"- bs=2 accuracy: `{payload['batched']['accuracy']:.8f}`",
        f"- accuracy delta: `{comparison['accuracy_delta']:.8f}`",
        f"- prediction flips: `{comparison['prediction_flip_count']}`",
        f"- max logit L_inf drift: `{comparison['max_logit_linf']:.8f}`",
        f"- bs=1 runtime_s: `{payload['single']['runtime_s']:.2f}`",
        f"- bs=2 runtime_s: `{payload['batched']['runtime_s']:.2f}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/experiment/public_mask_batching_20260605")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--softmax-config", default="scaled_k2_i8")
    parser.add_argument("--sqrt-method", default="MLFormer")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with crypten_ready():
        synthetic = run_synthetic(args.softmax_config, args.sqrt_method)
        payload = {
            "run_id": "public_mask_batching_20260605",
            "max_samples": args.max_samples,
            "synthetic": synthetic,
        }

        if not args.skip_e2e:
            model, tokenizer, dataset = load_sst2(args.max_samples)
            single = run_private_eval(
                model, tokenizer, dataset, 1, args.softmax_config, args.sqrt_method, masked=True
            )
            batched = run_private_eval(
                model, tokenizer, dataset, 2, args.softmax_config, args.sqrt_method, masked=True
            )
            comparison = compare_runs(single, batched)
            accepted = (
                abs(comparison["accuracy_delta"]) < 1e-12
                and comparison["prediction_flip_count"] == 0
                and synthetic["masked_pad_mass_max"] < 1e-8
            )
            payload.update(
                {
                    "single": single,
                    "batched": batched,
                    "comparison": comparison,
                    "accepted": accepted,
                }
            )
            write_summary(output_dir, payload)

    output_file = output_dir / "results.json"
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({k: payload[k] for k in payload if k != "single" and k != "batched"}, indent=2))
    print(f"output={output_file}")


if __name__ == "__main__":
    main()
