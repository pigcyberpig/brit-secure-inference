import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
TEXT_CLASSIFICATION_ROOT = str(get_data_root())

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import crypten as ct
import datasets
import torch
import transformers
from crypten.config import cfg
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--softmax-config", default="scaled_k2_i8")
    parser.add_argument("--sqrt-method", default="NR")
    return parser.parse_args()


def load_subset(indices):
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_error()

    raw = load_dataset("parquet", data_files={"validation": VALIDATION_FILE})["validation"]
    selected = raw.select(indices)

    config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    model.eval()

    def preprocess_function(examples):
        result = tokenizer(examples["sentence"], padding=False, max_length=MAX_LENGTH, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed = selected.map(preprocess_function, batched=True, remove_columns=selected.column_names)
    dataloader = DataLoader(processed, collate_fn=DataCollatorWithPadding(tokenizer), batch_size=1)
    return raw, selected, model, dataloader


def run_private(model, dataloader, indices, softmax_config, sqrt_method):
    ct.init()
    softmax_cfg = softmax_override(softmax_config)
    results = []

    with cfg.temp_override({"functions.sqrt_method": sqrt_method}):
        with cfg.temp_override(softmax_cfg):
            dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
            private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to("cuda:0")

            for idx, batch in zip(indices, dataloader):
                with ct.no_grad():
                    logits = private_model(
                        ct.cryptensor(batch["input_ids"]).to("cuda:0"),
                        ct.cryptensor(batch["attention_mask"]).to("cuda:0"),
                        ct.cryptensor(batch["token_type_ids"]).to("cuda:0"),
                    ).get_plain_text().cpu()[0]

                pred = int(logits.argmax().item())
                label = int(batch["labels"][0].item())
                results.append(
                    {
                        "idx": int(idx),
                        "pred": pred,
                        "label": label,
                        "correct": pred == label,
                        "logit_0": float(logits[0].item()),
                        "logit_1": float(logits[1].item()),
                    }
                )

    return results


def main():
    args = parse_args()
    source_rows = json.loads(Path(args.indices_file).read_text())
    indices = [int(row["idx"]) for row in source_rows]

    raw, selected, model, dataloader = load_subset(indices)
    results = run_private(model, dataloader, indices, args.softmax_config, args.sqrt_method)

    original_by_idx = {int(row["idx"]): row for row in source_rows}
    recovered = []
    still_wrong = []
    for row in results:
        merged = dict(row)
        merged["sentence"] = raw[row["idx"]]["sentence"]
        merged["previous"] = original_by_idx[row["idx"]]
        if row["correct"]:
            recovered.append(merged)
        else:
            still_wrong.append(merged)

    payload = {
        "softmax_config": args.softmax_config,
        "sqrt_method": args.sqrt_method,
        "count": len(results),
        "accuracy": sum(1 for row in results if row["correct"]) / len(results) if results else 0.0,
        "recovered_count": len(recovered),
        "still_wrong_count": len(still_wrong),
        "recovered_indices": [row["idx"] for row in recovered],
        "still_wrong_indices": [row["idx"] for row in still_wrong],
        "results": results,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"subset_count={len(results)}")
    print(f"accuracy={payload['accuracy']:.6f}")
    print(f"recovered={len(recovered)} still_wrong={len(still_wrong)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
