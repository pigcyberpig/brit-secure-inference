import argparse
import json
import os
import sys
from pathlib import Path
from types import MethodType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
TEXT_CLASSIFICATION_ROOT = str(get_data_root())

import crypten as ct
import torch
from crypten.config import cfg
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = f"{TEXT_CLASSIFICATION_ROOT}/bert-base-cased-sst2"
VALIDATION_FILE = f"{TEXT_CLASSIFICATION_ROOT}/glue/sst2/validation.parquet"
MAX_LENGTH = 128
OUTPUT_DIR = Path(os.path.join(QUEST_ROOT, "artifacts", "legacy_output", "trace_output"))


def softmax_override(name):
    if name == "ode_clip_i16":
        return {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": True,
            "functions.softmax_ode_iter_num": 16,
        }
    if name.startswith("scaled_k"):
        return {"functions.softmax_method": name}
    raise ValueError(f"unknown softmax config: {name}")


def capture_nodes():
    nodes = []
    for layer_idx in range(12):
        prefix = f"/bert/encoder/layer.{layer_idx}/attention/self"
        nodes.extend(
            [
                f"{prefix}/Add_output_0",
                f"{prefix}/Softmax_output_0",
            ]
        )
    return nodes


def load_model_and_tokenizer():
    config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    model.eval()
    return model, tokenizer


def load_pair(indices):
    raw = load_dataset("parquet", data_files={"validation": VALIDATION_FILE})["validation"]
    samples = [raw[idx] for idx in indices]
    return samples


def tokenize_pair(tokenizer, samples):
    sentences = [sample["sentence"] for sample in samples]
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    lengths = encoded["attention_mask"].sum(dim=-1).tolist()
    return encoded, lengths


def summarize_pre_softmax(sample_tensor, valid_len):
    total_len = sample_tensor.shape[-1]
    query_len = min(valid_len, sample_tensor.shape[-2])
    valid_queries = sample_tensor[:, :query_len, :]
    row_mean = valid_queries.mean(dim=-1, keepdim=True)
    centered = valid_queries - row_mean

    summary = {
        "valid_len": int(valid_len),
        "total_len": int(total_len),
        "pad_tokens": int(total_len - valid_len),
        "overall_mean": float(valid_queries.mean().item()),
        "overall_min": float(valid_queries.min().item()),
        "overall_max": float(valid_queries.max().item()),
    }

    valid_keys = valid_queries[..., :valid_len]
    summary.update(
        {
            "valid_key_mean": float(valid_keys.mean().item()),
            "valid_key_min": float(valid_keys.min().item()),
            "valid_key_max": float(valid_keys.max().item()),
            "centered_valid_key_mean": float(centered[..., :valid_len].mean().item()),
            "centered_valid_key_max_abs": float(centered[..., :valid_len].abs().max().item()),
        }
    )

    if valid_len < total_len:
        pad_keys = valid_queries[..., valid_len:]
        summary.update(
            {
                "pad_key_mean": float(pad_keys.mean().item()),
                "pad_key_min": float(pad_keys.min().item()),
                "pad_key_max": float(pad_keys.max().item()),
                "centered_pad_key_mean": float(centered[..., valid_len:].mean().item()),
                "centered_pad_key_max_abs": float(centered[..., valid_len:].abs().max().item()),
            }
        )
    return summary


def summarize_softmax(sample_tensor, valid_len):
    total_len = sample_tensor.shape[-1]
    query_len = min(valid_len, sample_tensor.shape[-2])
    valid_queries = sample_tensor[:, :query_len, :]
    summary = {
        "valid_len": int(valid_len),
        "total_len": int(total_len),
        "pad_tokens": int(total_len - valid_len),
        "max_prob": float(valid_queries.max().item()),
        "min_prob": float(valid_queries.min().item()),
        "row_sum_max_abs_err": float((valid_queries.sum(dim=-1) - 1.0).abs().max().item()),
    }

    valid_mass = valid_queries[..., :valid_len].sum(dim=-1)
    summary["valid_mass_mean"] = float(valid_mass.mean().item())
    summary["valid_mass_min"] = float(valid_mass.min().item())

    if valid_len < total_len:
        pad_mass = valid_queries[..., valid_len:].sum(dim=-1)
        summary.update(
            {
                "pad_mass_mean": float(pad_mass.mean().item()),
                "pad_mass_max": float(pad_mass.max().item()),
                "pad_mass_min": float(pad_mass.min().item()),
            }
        )
    return summary


class NodeCollector:
    def __init__(self, node_names):
        self.node_names = node_names
        self.current = None

    def start(self):
        self.current = {}

    def finish(self):
        result = self.current or {}
        self.current = None
        return result

    def install(self, private_model):
        for node_name in self.node_names:
            module = private_model._modules[node_name]
            original_forward = object.__getattribute__(module, "forward")

            def wrapped_forward(module_self, *args, _orig=original_forward, _name=node_name, **kwargs):
                output = _orig(*args, **kwargs)
                if self.current is not None:
                    self.current[_name] = output.get_plain_text().float().cpu()
                return output

            module.forward = MethodType(wrapped_forward, module)


def run_config(softmax_config, sqrt_method, encoded, lengths, node_names):
    model, _ = load_model_and_tokenizer()
    collector = NodeCollector(node_names)
    with cfg.temp_override({"functions.sqrt_method": sqrt_method}):
        with cfg.temp_override(softmax_override(softmax_config)):
            dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
            private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to("cuda:0")
            collector.install(private_model)
            collector.start()
            with ct.no_grad():
                logits = private_model(
                    ct.cryptensor(encoded["input_ids"]).to("cuda:0"),
                    ct.cryptensor(encoded["attention_mask"]).to("cuda:0"),
                    ct.cryptensor(encoded["token_type_ids"]).to("cuda:0"),
                ).get_plain_text().cpu()
            tensors = collector.finish()

    traces = {}
    for layer_idx in range(12):
        add_name = f"/bert/encoder/layer.{layer_idx}/attention/self/Add_output_0"
        softmax_name = f"/bert/encoder/layer.{layer_idx}/attention/self/Softmax_output_0"
        traces[f"layer_{layer_idx}"] = []
        for sample_pos, valid_len in enumerate(lengths):
            traces[f"layer_{layer_idx}"].append(
                {
                    "sample_pos": sample_pos,
                    "pre_softmax": summarize_pre_softmax(tensors[add_name][sample_pos], valid_len),
                    "softmax": summarize_softmax(tensors[softmax_name][sample_pos], valid_len),
                }
            )

    del private_model
    del model
    torch.cuda.empty_cache()
    return {
        "logits": logits.tolist(),
        "preds": logits.argmax(dim=-1).tolist(),
        "layers": traces,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, type=int, required=True)
    parser.add_argument("--sqrt_method", default="NR")
    parser.add_argument("--baseline_softmax", default="ode_clip_i16")
    parser.add_argument("--scaled_softmax", default="scaled_k2_i8")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ct.init()

    samples = load_pair(args.pair)
    _, tokenizer = load_model_and_tokenizer()
    encoded, lengths = tokenize_pair(tokenizer, samples)
    node_names = capture_nodes()

    baseline = run_config(args.baseline_softmax, args.sqrt_method, encoded, lengths, node_names)
    scaled = run_config(args.scaled_softmax, args.sqrt_method, encoded, lengths, node_names)

    report = {
        "pair": args.pair,
        "lengths": lengths,
        "labels": [int(sample["label"]) for sample in samples],
        "sentences": [sample["sentence"] for sample in samples],
        "baseline_softmax": args.baseline_softmax,
        "scaled_softmax": args.scaled_softmax,
        "baseline": baseline,
        "scaled": scaled,
    }

    for layer_idx in range(12):
        print(f"layer {layer_idx}")
        for sample_pos in range(len(samples)):
            base_soft = baseline["layers"][f"layer_{layer_idx}"][sample_pos]["softmax"]
            scaled_soft = scaled["layers"][f"layer_{layer_idx}"][sample_pos]["softmax"]
            base_pre = baseline["layers"][f"layer_{layer_idx}"][sample_pos]["pre_softmax"]
            scaled_pre = scaled["layers"][f"layer_{layer_idx}"][sample_pos]["pre_softmax"]
            print(
                f"  sample_pos={sample_pos} len={lengths[sample_pos]} "
                f"pad={base_soft['pad_tokens']} logits_base={baseline['logits'][sample_pos]} "
                f"logits_scaled={scaled['logits'][sample_pos]}"
            )
            print(
                f"    pre centered_valid_max_abs base/scaled="
                f"{base_pre['centered_valid_key_max_abs']:.3g}/{scaled_pre['centered_valid_key_max_abs']:.3g}"
            )
            if "pad_key_mean" in base_pre:
                print(
                    f"    pre centered_pad_mean base/scaled="
                    f"{base_pre['centered_pad_key_mean']:.3g}/{scaled_pre['centered_pad_key_mean']:.3g}"
                )
            if "pad_mass_mean" in base_soft:
                print(
                    f"    softmax pad_mass_mean base/scaled="
                    f"{base_soft['pad_mass_mean']:.3g}/{scaled_soft['pad_mass_mean']:.3g} "
                    f"pad_mass_max base/scaled={base_soft['pad_mass_max']:.3g}/{scaled_soft['pad_mass_max']:.3g}"
                )

    output_file = OUTPUT_DIR / f"padding_pair_{args.pair[0]}_{args.pair[1]}_{args.scaled_softmax}_{args.sqrt_method}.json"
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved padding analysis to {output_file}")


if __name__ == "__main__":
    main()
