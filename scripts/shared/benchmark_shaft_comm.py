import argparse
import time

import torch

import crypten
import crypten.communicator as comm


STANDARD_CONFIG = {
    "softmax": {"functions.softmax_method": "reciprocal"},
    "gelu": {"functions.gelu_method": "erf"},
}

SHAFT_CONFIG = {
    "softmax": {"functions.softmax_method": "ode"},
    "gelu": {"functions.gelu_method": "fourier"},
}


def _input_tensors(rows, cols):
    torch.manual_seed(0)
    softmax_input = torch.randn(rows, cols)
    gelu_input = torch.linspace(-4.0, 4.0, steps=rows * cols).reshape(rows, cols)
    return softmax_input, gelu_input


def _avg_dict(items):
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def _measure_op(name, encrypted_input, op, repeats):
    measurements = []
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        op(encrypted_input)
        elapsed = time.perf_counter() - start
        stats = crypten.get_communication_stats()
        measurements.append(
            {
                "wall_time_s": elapsed,
                "comm_time_s": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )
    result = _avg_dict(measurements)
    result["name"] = name
    return result


def _run_rank(rows, cols, repeats):
    softmax_plain, gelu_plain = _input_tensors(rows, cols)
    softmax_enc = crypten.cryptensor(softmax_plain, src=0)
    gelu_enc = crypten.cryptensor(gelu_plain, src=0)

    rank_results = []
    cases = [
        ("crypten_standard", STANDARD_CONFIG),
        ("shaft", SHAFT_CONFIG),
    ]

    for case_name, config in cases:
        with crypten.cfg.temp_override(config["softmax"]):
            rank_results.append(
                _measure_op(
                    f"{case_name}.softmax",
                    softmax_enc,
                    lambda x: x.softmax(dim=-1),
                    repeats,
                )
            )

        with crypten.cfg.temp_override(config["gelu"]):
            rank_results.append(
                _measure_op(
                    f"{case_name}.gelu",
                    gelu_enc,
                    lambda x: x.gelu(),
                    repeats,
                )
            )

    return {
        "rank": comm.get().get_rank(),
        "world_size": comm.get().get_world_size(),
        "rows": rows,
        "cols": cols,
        "repeats": repeats,
        "results": rank_results,
    }


@crypten.mpc.run_multiprocess(world_size=2)
def _run_multiprocess(rows, cols, repeats):
    return _run_rank(rows, cols, repeats)


def _ratio(new_value, base_value):
    if base_value == 0:
        return float("inf") if new_value else 1.0
    return new_value / base_value


def _print_summary(rank_outputs):
    by_name = {}
    for rank_output in rank_outputs:
        for result in rank_output["results"]:
            by_name.setdefault(result["name"], []).append(result)

    rows = rank_outputs[0]["rows"]
    cols = rank_outputs[0]["cols"]
    repeats = rank_outputs[0]["repeats"]
    print(f"world_size=2 shape=({rows}, {cols}) repeats={repeats}")
    print(
        "metric columns: max wall latency, max comm latency, system bytes=sum(rank bytes)/2, max rounds"
    )
    print(
        f"{'case':<25} {'wall_ms':>12} {'comm_ms':>12} "
        f"{'system_bytes':>14} {'rounds':>8}"
    )

    summary = {}
    for name, results in sorted(by_name.items()):
        wall_time = max(result["wall_time_s"] for result in results)
        comm_time = max(result["comm_time_s"] for result in results)
        system_bytes = sum(result["bytes"] for result in results) / 2
        rounds = max(result["rounds"] for result in results)
        summary[name] = {
            "wall_time_s": wall_time,
            "comm_time_s": comm_time,
            "system_bytes": system_bytes,
            "rounds": rounds,
        }
        print(
            f"{name:<25} {wall_time * 1000:12.3f} {comm_time * 1000:12.3f} "
            f"{system_bytes:14.0f} {rounds:8.1f}"
        )

    print("\nSHAFT / CrypTen standard ratios")
    for op_name in ["softmax", "gelu"]:
        base = summary[f"crypten_standard.{op_name}"]
        shaft = summary[f"shaft.{op_name}"]
        print(
            f"{op_name}: "
            f"wall={_ratio(shaft['wall_time_s'], base['wall_time_s']):.3f}x, "
            f"comm_time={_ratio(shaft['comm_time_s'], base['comm_time_s']):.3f}x, "
            f"bytes={_ratio(shaft['system_bytes'], base['system_bytes']):.3f}x, "
            f"rounds={_ratio(shaft['rounds'], base['rounds']):.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    outputs = _run_multiprocess(args.rows, args.cols, args.repeats)
    _print_summary(outputs)


if __name__ == "__main__":
    main()
