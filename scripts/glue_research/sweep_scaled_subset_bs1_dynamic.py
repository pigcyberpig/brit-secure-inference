import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QUEST_ROOT, get_shaft_root, get_data_root  # noqa: E402

SHAFT_ROOT = str(get_shaft_root())
TEXT_CLASSIFICATION_ROOT = str(get_data_root())

from eval_scaled_subset_bs1_dynamic import load_subset, run_private  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--sqrt-method", default="NR")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "scaled_k2_i8",
            "scaled_k2_i12",
            "scaled_k2_i16",
            "scaled_k3_i8",
            "scaled_k3_i12",
            "scaled_k4_i8",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_rows = json.loads(Path(args.indices_file).read_text())
    indices = [int(row["idx"]) for row in source_rows]
    raw, selected, model, dataloader = load_subset(indices)

    summary_rows = []
    details = {}
    for config_name in args.configs:
        print(f"running {config_name}")
        results = run_private(model, dataloader, indices, config_name, args.sqrt_method)
        recovered = [row["idx"] for row in results if row["correct"]]
        still_wrong = [row["idx"] for row in results if not row["correct"]]
        accuracy = len(recovered) / len(results) if results else 0.0
        summary_rows.append(
            {
                "softmax_config": config_name,
                "accuracy_on_prev_wrong_subset": accuracy,
                "recovered_count": len(recovered),
                "still_wrong_count": len(still_wrong),
            }
        )
        details[config_name] = {
            "results": results,
            "recovered_indices": recovered,
            "still_wrong_indices": still_wrong,
        }
        print(
            f"{config_name}: accuracy={accuracy:.6f} "
            f"recovered={len(recovered)} still_wrong={len(still_wrong)}"
        )

    payload = {
        "subset_size": len(indices),
        "sqrt_method": args.sqrt_method,
        "configs": args.configs,
        "summary": summary_rows,
        "details": details,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
