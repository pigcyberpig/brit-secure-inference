
import argparse
import json
import logging
import math
import os
import time

# Stable default: both MPC ranks share GPU0, matching prior full runs.
# Set CUDA_VISIBLE_DEVICES=0,1 and --cuda_device -1 explicitly for dual-GPU tests.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import datasets
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

from transformers.utils import check_min_version
try:
    from transformers.utils import send_example_telemetry
except ImportError:
    def send_example_telemetry(*args, **kwargs):
        return None
from transformers.utils.versions import require_version

import crypten as ct
import crypten.communicator as comm
from crypten.config import cfg
from multiprocess_launcher import MultiProcessLauncher

# This quest runs against a local SHAFT stack pinned to an older Transformers build.
# Keep the example runnable under that environment instead of enforcing a newer example-only minimum.
check_min_version("4.20.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def validate_softmax_config(name):
    if name == "ode_clip_i16":
        return name
    if name.startswith("scaled_k") and "_i" in name:
        scale_part, iter_part = name[len("scaled_k") :].split("_i", 1)
        if (
            scale_part.isdigit()
            and iter_part.isdigit()
            and int(scale_part) > 0
            and int(iter_part) > 0
        ):
            return name
    raise argparse.ArgumentTypeError(
        "Softmax configuration must be `ode_clip_i16` or `scaled_kK_iN`."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='qnli',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv, json, or parquet file containing the validation data.",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=-1,
        help="Number of validation data to run, set to -1 if run the whole dataset.",
    )
    parser.add_argument(
        "--len_data",
        type=int,
        default=128,
        help="Sequence length of data to run, set to -1 if run the whole dataset.",
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="If passed, estimate computation time (without communication).",
    )
    parser.add_argument(
        "--acc",
        action="store_true",
        help="If passed, evaluate private inference accuracy on the entire dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-cased-qnli',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default='eval_private/qnli/', help="Where to store the output.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--softmax_config",
        type=validate_softmax_config,
        default="ode_clip_i16",
        help="Softmax configuration for private inference.",
    )
    parser.add_argument(
        "--sqrt_method",
        type=str,
        choices=["NR", "MLFormer"],
        default="NR",
        help="Inverse square root method used by LayerNorm.",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help=(
            "CUDA device index to use inside each process. "
            "Use -1 to auto-assign one visible GPU per MPC rank."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Stop after this many evaluated samples after any len_data filter.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.validation_file is None:
        raise ValueError("Need either a task name or a validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "parquet"], (
                "`validation_file` should be a csv, json, or parquet file."
            )

    return args


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


def pearson_corr(predictions, references):
    n = len(predictions)
    if n == 0:
        return 0.0
    pred_mean = sum(predictions) / n
    ref_mean = sum(references) / n
    numerator = sum((p - pred_mean) * (r - ref_mean) for p, r in zip(predictions, references))
    pred_var = sum((p - pred_mean) ** 2 for p in predictions)
    ref_var = sum((r - ref_mean) ** 2 for r in references)
    denominator = math.sqrt(pred_var * ref_var)
    return numerator / denominator if denominator else 0.0


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_corr(predictions, references):
    if not predictions:
        return 0.0
    return pearson_corr(average_ranks(predictions), average_ranks(references))


def matthews_corrcoef(predictions, references):
    labels = sorted(set(predictions) | set(references))
    if len(labels) <= 1:
        return 0.0
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for pred, ref in zip(predictions, references):
        matrix[label_to_idx[ref]][label_to_idx[pred]] += 1

    total = sum(sum(row) for row in matrix)
    trace = sum(matrix[i][i] for i in range(len(labels)))
    pred_counts = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    true_counts = [sum(matrix[i]) for i in range(len(labels))]
    numerator = trace * total - sum(p * t for p, t in zip(pred_counts, true_counts))
    denominator = math.sqrt(
        (total**2 - sum(p**2 for p in pred_counts))
        * (total**2 - sum(t**2 for t in true_counts))
    )
    return numerator / denominator if denominator else 0.0


def compute_metrics(task_name, predictions, references, is_regression):
    if is_regression:
        pearson = pearson_corr(predictions, references)
        spearman = spearman_corr(predictions, references)
        return {
            "pearson": pearson,
            "spearmanr": spearman,
            "combined_score": (pearson + spearman) / 2.0,
        }

    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    metrics = {"accuracy": correct / len(references) if references else 0.0}
    if task_name == "cola":
        metrics["matthews_correlation"] = matthews_corrcoef(predictions, references)
    return metrics


def main():
    script_start_time = time.time()
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_private", args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if args.validation_file is None and args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        #  raw_datasets = load_dataset("glue", args.task_name)
        raw_datasets = load_dataset(
            "parquet",
            data_files={
                "train": f"glue/{args.task_name}/train.parquet",
                "validation": f"glue/{args.task_name}/validation.parquet",
                "test": f"glue/{args.task_name}/test.parquet"
            }
        )
    else:
        # Loading the dataset from a local validation file.
        data_files = {}
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets[validation_key].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets[validation_key].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets[validation_key].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets[validation_key].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
    
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # 处理COLA任务可能的空标签（不影响其他任务）
                if args.task_name == "cola":
                    result["labels"] = [label_to_id.get(l, -1) for l in examples["label"] if l is not None]
                # 保留RTE任务特殊处理逻辑
                elif args.task_name == "rte":
                    processed_labels = []
                    for l in examples["label"]:
                        l_str = str(l)
                        l_clean = 0 if l == -1 else int(l_str)
                        if l_clean not in label_to_id and l_str in label_to_id:
                            l_clean = label_to_id[l_str]
                        processed_labels.append(l_clean)
                    result["labels"] = [label_to_id.get(l, 0) for l in processed_labels]
                # 其他任务（如QNLI）保持原逻辑
                else:
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # 无label_to_id时直接赋值（兼容回归任务）
                result["labels"] = examples["label"]
        return result

    remove_columns = (
        raw_datasets["train"].column_names
        if "train" in raw_datasets
        else raw_datasets[validation_key].column_names
    )

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    metric_predictions = []
    metric_references = []

    device = "cpu"
    if torch.cuda.is_available():
        cuda_device = args.cuda_device
        if cuda_device < 0:
            cuda_device = comm.get().get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(cuda_device)
        device = f"cuda:{cuda_device}"
    ct.init()
    is_main_process = comm.get().get_rank() == 0
    softmax_cfg = softmax_override(args.softmax_config)
    
    with cfg.temp_override({"functions.sqrt_method": args.sqrt_method}):
        with cfg.temp_override(softmax_cfg):
            dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
            private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
            model.eval()
            samples_seen = 0
            private_forward_stats = {
                "total_time_s": 0.0,
                "total_comm_bytes": 0,
                "total_comm_rounds": 0,
                "total_comm_time_s": 0.0,
                "embedding_time_s": 0.0,
                "embedding_comm_bytes": 0,
                "embedding_comm_rounds": 0,
                "embedding_comm_time_s": 0.0,
                "matmul_time_s": 0.0,
                "matmul_comm_bytes": 0,
                "matmul_comm_rounds": 0,
                "matmul_comm_time_s": 0.0,
                "softmax_time_s": 0.0,
                "softmax_comm_bytes": 0,
                "softmax_comm_rounds": 0,
                "softmax_comm_time_s": 0.0,
                "gelu_time_s": 0.0,
                "gelu_comm_bytes": 0,
                "gelu_comm_rounds": 0,
                "gelu_comm_time_s": 0.0,
                "layernorm_time_s": 0.0,
                "layernorm_comm_bytes": 0,
                "layernorm_comm_rounds": 0,
                "layernorm_comm_time_s": 0.0,
                "tanh_time_s": 0.0,
                "tanh_comm_bytes": 0,
                "tanh_comm_rounds": 0,
                "tanh_comm_time_s": 0.0,
                "conv_time_s": 0.0,
                "conv_comm_bytes": 0,
                "conv_comm_rounds": 0,
                "conv_comm_time_s": 0.0,
                "other_time_s": 0.0,
                "other_comm_bytes": 0,
                "other_comm_rounds": 0,
                "other_comm_time_s": 0.0,
            }
            loop_start_time = time.time()
            ct.reset_communication_stats()
            for step, batch in enumerate(eval_dataloader):
                if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
                    continue

                inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
                attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
                token_type_enc = ct.cryptensor(batch["token_type_ids"]).to(device)

                with ct.no_grad():
                    forward_start_time = time.time()
                    outputs_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)
                private_forward_stats["total_time_s"] += time.time() - forward_start_time
                private_forward_stats["total_comm_bytes"] += int(getattr(private_model, "total_comm_bytes", 0))
                private_forward_stats["total_comm_rounds"] += int(getattr(private_model, "total_comm_rounds", 0))
                private_forward_stats["total_comm_time_s"] += float(getattr(private_model, "total_comm_time", 0.0))
                for module_name in (
                    "embedding",
                    "matmul",
                    "softmax",
                    "gelu",
                    "layernorm",
                    "tanh",
                    "conv",
                    "other",
                ):
                    private_forward_stats[f"{module_name}_time_s"] += float(
                        getattr(private_model, f"{module_name}_time", 0.0)
                    )
                    private_forward_stats[f"{module_name}_comm_bytes"] += int(
                        getattr(private_model, f"{module_name}_comm_bytes", 0)
                    )
                    private_forward_stats[f"{module_name}_comm_rounds"] += int(
                        getattr(private_model, f"{module_name}_comm_rounds", 0)
                    )
                    private_forward_stats[f"{module_name}_comm_time_s"] += float(
                        getattr(private_model, f"{module_name}_comm_time", 0.0)
                    )

                outputs = outputs_enc.get_plain_text().cpu()

                predictions = outputs.argmax(dim=-1) if not is_regression else outputs.squeeze(-1)
                predictions, references = predictions, batch["labels"]

                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                samples_seen += references.shape[0]
                if is_regression:
                    metric_predictions.extend(float(value) for value in predictions.tolist())
                    metric_references.extend(float(value) for value in references.tolist())
                else:
                    metric_predictions.extend(int(value) for value in predictions.tolist())
                    metric_references.extend(int(value) for value in references.tolist())
                if args.max_samples is not None and samples_seen >= args.max_samples:
                    break
                elif args.num_data > 0 and samples_seen >= args.num_data:
                    break
                print(f"running private inference, samples_seen={samples_seen}")
    
    try:
        eval_metric = compute_metrics(args.task_name, metric_predictions, metric_references, is_regression)
        elapsed = time.time() - loop_start_time
        stats = ct.get_communication_stats() if comm.get().get_world_size() > 1 else None
        result_payload = {
            "softmax_config": args.softmax_config,
            "sqrt_method": args.sqrt_method,
            "metric": eval_metric,
            "running_time_s": elapsed,
            "samples_seen": samples_seen,
            "private_forward": private_forward_stats,
        }
        if stats is not None:
            result_payload["communication"] = stats
        if is_main_process:
            print(f"metric: {eval_metric}")
            print(f"running time: {elapsed}s")
            print(f"private forward: {private_forward_stats}")
            if stats is not None:
                print(f"communication: {stats}")

        if is_main_process and args.output_dir is not None:
            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)
            with open(os.path.join(args.output_dir, f"summary_{args.softmax_config}.json"), "w") as f:
                json.dump(result_payload, f)
    except Exception as exc:
        if comm.get().get_rank() == 0:
            print(f"evaluation failed: {exc}")
        raise

if __name__ == "__main__":
    args = parse_args()
    if args.comp:
        # run without communication
        with cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comp"}):
            main()
    elif args.acc:
        # run without communication and cost printing
        with cfg.temp_override({"cost.estimate_cost": False}):
            main()
    else:
        # run with communication
        launcher = MultiProcessLauncher(2, main)
        launcher.start()
        launcher.join()
        launcher.terminate()
