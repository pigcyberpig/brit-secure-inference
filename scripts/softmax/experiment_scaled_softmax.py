import argparse
import time

import torch

import crypten
import crypten.communicator as comm


def _make_input(rows, cols, input_scale):
    torch.manual_seed(0)
    base = torch.randn(rows, cols)
    trend = torch.linspace(-1.0, 1.0, steps=cols).reshape(1, cols)
    return input_scale * (base + trend)


def _pow_int(tensor, exponent):
    if exponent < 1:
        raise ValueError("exponent must be positive")
    result = None
    base = tensor
    current = exponent
    while current:
        if current & 1:
            result = base if result is None else result * base
        current >>= 1
        if current:
            base = base * base
    return result


def scaled_ode_softmax(tensor, dim, scale):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / scale
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
        }
    ):
        probs = scaled.softmax(dim=dim)
    powered = _pow_int(probs, scale)
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_total = powered.sum(dim=dim, keepdim=True).reciprocal()
    return powered * inv_total


def _metrics(output, reference):
    diff = (output - reference).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "row_sum_max_abs": (output.sum(dim=-1) - 1).abs().max().item(),
    }


def _measure(name, encrypted_input, reference, fn, repeats):
    measurements = []
    last_output = None
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        encrypted_output = fn(encrypted_input)
        wall_time = time.perf_counter() - start
        stats = crypten.get_communication_stats()
        output = encrypted_output.get_plain_text()
        last_output = output
        measurements.append(
            {
                "wall_time_s": wall_time,
                "comm_time_s": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )

    averaged = {
        key: sum(item[key] for item in measurements) / len(measurements)
        for key in measurements[0]
    }
    averaged.update(_metrics(last_output, reference))
    averaged["name"] = name
    return averaged


def _run_rank(rows, cols, input_scale, repeats, scales):
    plain = _make_input(rows, cols, input_scale)
    reference = torch.softmax(plain, dim=-1)
    encrypted = crypten.cryptensor(plain, src=0)

    results = []

    with crypten.cfg.temp_override({"functions.softmax_method": "reciprocal"}):
        results.append(
            _measure(
                "crypten_standard_reciprocal",
                encrypted,
                reference,
                lambda x: x.softmax(dim=-1),
                repeats,
            )
        )

    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": True,
        }
    ):
        results.append(
            _measure(
                "shaft_ode_clip",
                encrypted,
                reference,
                lambda x: x.softmax(dim=-1),
                repeats,
            )
        )

    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
        }
    ):
        results.append(
            _measure(
                "shaft_ode_no_clip",
                encrypted,
                reference,
                lambda x: x.softmax(dim=-1),
                repeats,
            )
        )

    for scale in scales:
        results.append(
            _measure(
                f"scaled_center_recover_k{scale}",
                encrypted,
                reference,
                lambda x, scale=scale: scaled_ode_softmax(x, dim=-1, scale=scale),
                repeats,
            )
        )

    return {
        "rank": comm.get().get_rank(),
        "world_size": comm.get().get_world_size(),
        "rows": rows,
        "cols": cols,
        "input_scale": input_scale,
        "repeats": repeats,
        "results": results,
    }


@crypten.mpc.run_multiprocess(world_size=2)
def _run_multiprocess(rows, cols, input_scale, repeats, scales):
    return _run_rank(rows, cols, input_scale, repeats, scales)


def _summarize(outputs):
    by_name = {}
    for output in outputs:
        for result in output["results"]:
            by_name.setdefault(result["name"], []).append(result)

    meta = outputs[0]
    print(
        f"world_size=2 shape=({meta['rows']}, {meta['cols']}) "
        f"input_scale={meta['input_scale']} repeats={meta['repeats']}"
    )
    print(
        f"{'case':<30} {'wall_ms':>10} {'comm_ms':>10} {'sys_bytes':>12} "
        f"{'rounds':>8} {'max_abs':>10} {'mean_abs':>10} {'row_sum':>10}"
    )

    summary = {}
    for name, rank_results in sorted(by_name.items()):
        item = {
            "wall_time_s": max(result["wall_time_s"] for result in rank_results),
            "comm_time_s": max(result["comm_time_s"] for result in rank_results),
            "system_bytes": sum(result["bytes"] for result in rank_results) / 2,
            "rounds": max(result["rounds"] for result in rank_results),
            "max_abs": max(result["max_abs"] for result in rank_results),
            "mean_abs": max(result["mean_abs"] for result in rank_results),
            "row_sum_max_abs": max(result["row_sum_max_abs"] for result in rank_results),
        }
        summary[name] = item
        print(
            f"{name:<30} {item['wall_time_s'] * 1000:10.3f} "
            f"{item['comm_time_s'] * 1000:10.3f} {item['system_bytes']:12.0f} "
            f"{item['rounds']:8.1f} {item['max_abs']:10.6f} "
            f"{item['mean_abs']:10.6f} {item['row_sum_max_abs']:10.6f}"
        )

    base = summary["shaft_ode_clip"]
    print("\nRatios vs shaft_ode_clip")
    for name, item in summary.items():
        if name == "shaft_ode_clip":
            continue
        print(
            f"{name}: wall={item['wall_time_s'] / base['wall_time_s']:.3f}x, "
            f"comm={item['comm_time_s'] / base['comm_time_s']:.3f}x, "
            f"bytes={item['system_bytes'] / base['system_bytes']:.3f}x, "
            f"rounds={item['rounds'] / base['rounds']:.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--input-scale", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()

    outputs = _run_multiprocess(
        args.rows,
        args.cols,
        args.input_scale,
        args.repeats,
        args.scales,
    )
    _summarize(outputs)


if __name__ == "__main__":
    main()
