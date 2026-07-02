import argparse
import time

import torch

import crypten
import crypten.communicator as comm


LAN_BPS = 1_000_000_000
LAN_LATENCY_S = 0.0005
WAN_BPS = 400_000_000
WAN_LATENCY_S = 0.004


def make_input(rows, cols, input_scale):
    torch.manual_seed(0)
    base = torch.randn(rows, cols)
    trend = torch.linspace(-1.0, 1.0, steps=cols).reshape(1, cols)
    return input_scale * (base + trend)


def pow_int(tensor, exponent):
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


def scaled_k2_softmax(tensor, dim, iters):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / 2
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        probs = scaled.softmax(dim=dim)
    probs2 = probs * probs
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_total = probs2.sum(dim=dim, keepdim=True).reciprocal()
    return probs2 * inv_total


def ode_clip_i16(tensor, dim):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": True,
            "functions.softmax_ode_iter_num": 16,
        }
    ):
        return tensor.softmax(dim=dim)


def ode_no_clip_i16(tensor, dim):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
            "functions.softmax_ode_iter_num": 16,
        }
    ):
        return tensor.softmax(dim=dim)


def measure(encrypted, reference, fn, repeats):
    records = []
    last_output = None
    for _ in range(repeats):
        comm.get().barrier()
        crypten.reset_communication_stats()
        start = time.perf_counter()
        result = fn(encrypted)
        wall = time.perf_counter() - start
        stats = crypten.get_communication_stats()
        last_output = result.get_plain_text()
        records.append(
            {
                "wall": wall,
                "comm_time": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )

    diff = (last_output - reference).abs()
    averaged = {
        key: sum(record[key] for record in records) / len(records)
        for key in records[0]
    }
    averaged["compute"] = max(averaged["wall"] - averaged["comm_time"], 0.0)
    averaged["max_abs"] = diff.max().item()
    averaged["mean_abs"] = diff.mean().item()
    averaged["row_sum"] = (last_output.sum(dim=-1) - 1).abs().max().item()
    return averaged


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(rows, cols, input_scale, repeats):
    plain = make_input(rows, cols, input_scale)
    reference = torch.softmax(plain, dim=-1)
    encrypted = crypten.cryptensor(plain, src=0)
    cases = [
        ("ode_clip_i16", lambda x: ode_clip_i16(x, -1)),
        ("ode_no_clip_i16", lambda x: ode_no_clip_i16(x, -1)),
        ("scaled_k2_i8", lambda x: scaled_k2_softmax(x, -1, 8)),
        ("scaled_k2_i4", lambda x: scaled_k2_softmax(x, -1, 4)),
    ]
    return {
        "rank": comm.get().get_rank(),
        "rows": rows,
        "cols": cols,
        "input_scale": input_scale,
        "repeats": repeats,
        "results": [(name, measure(encrypted, reference, fn, repeats)) for name, fn in cases],
    }


def estimate_time(compute_s, comm_bytes, rounds, bandwidth_bps, latency_s):
    bytes_per_s = bandwidth_bps / 8
    return compute_s + 2 * comm_bytes / bytes_per_s + rounds * latency_s


def summarize(outputs):
    by_name = {}
    for output in outputs:
        for name, result in output["results"]:
            by_name.setdefault(name, []).append(result)

    meta = outputs[0]
    print(
        f"world_size=2 shape=({meta['rows']}, {meta['cols']}) "
        f"input_scale={meta['input_scale']} repeats={meta['repeats']}"
    )
    print(
        "paper_est_time = compute_time + 2*communication/bandwidth + rounds*latency"
    )
    print(
        f"{'case':<16} {'comp_ms':>9} {'comm_MB':>9} {'rounds':>7} "
        f"{'LAN_ms':>9} {'WAN_ms':>9} {'max_abs':>9} {'mean_abs':>9}"
    )
    for name, per_rank in by_name.items():
        compute = max(item["compute"] for item in per_rank)
        comm_bytes = sum(item["bytes"] for item in per_rank) / 2
        rounds = max(item["rounds"] for item in per_rank)
        lan = estimate_time(compute, comm_bytes, rounds, LAN_BPS, LAN_LATENCY_S)
        wan = estimate_time(compute, comm_bytes, rounds, WAN_BPS, WAN_LATENCY_S)
        max_abs = max(item["max_abs"] for item in per_rank)
        mean_abs = max(item["mean_abs"] for item in per_rank)
        print(
            f"{name:<16} {compute * 1000:9.3f} {comm_bytes / 1_000_000:9.4f} "
            f"{rounds:7.1f} {lan * 1000:9.3f} {wan * 1000:9.3f} "
            f"{max_abs:9.5f} {mean_abs:9.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    for cols in args.cols:
        outputs = run_rank(args.rows, cols, args.input_scale, args.repeats)
        summarize(outputs)


if __name__ == "__main__":
    main()
