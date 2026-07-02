import argparse
import time

import torch

import crypten
import crypten.communicator as comm


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


def scaled_recover_softmax(tensor, dim, scale, iters):
    centered = tensor - tensor.mean(dim=dim, keepdim=True)
    scaled = centered / scale
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": False,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        probs = scaled.softmax(dim=dim)
    powered = pow_int(probs, scale)
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        return powered * powered.sum(dim=dim, keepdim=True).reciprocal()


def ode_softmax(tensor, dim, iters, clip):
    with crypten.cfg.temp_override(
        {
            "functions.softmax_method": "ode",
            "functions.softmax_ode_clip": clip,
            "functions.softmax_ode_iter_num": iters,
        }
    ):
        return tensor.softmax(dim=dim)


def build_cases(scales, clip_iters, scaled_iters):
    cases = [
        (
            f"ode_clip_i{clip_iters}",
            lambda x, clip_iters=clip_iters: ode_softmax(x, -1, clip_iters, True),
        )
    ]
    for iters in scaled_iters:
        for scale in scales:
            cases.append(
                (
                    f"scaled_k{scale}_i{iters}",
                    lambda x, scale=scale, iters=iters: scaled_recover_softmax(
                        x, -1, scale, iters
                    ),
                )
            )
    return cases


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
                "comm": stats["time"],
                "bytes": stats["bytes"],
                "rounds": stats["rounds"],
            }
        )

    diff = (last_output - reference).abs()
    return {
        "wall": sum(r["wall"] for r in records) / repeats,
        "comm": sum(r["comm"] for r in records) / repeats,
        "bytes": sum(r["bytes"] for r in records) / repeats,
        "rounds": sum(r["rounds"] for r in records) / repeats,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "row_sum": (last_output.sum(dim=-1) - 1).abs().max().item(),
    }


@crypten.mpc.run_multiprocess(world_size=2)
def run_rank(rows, cols, input_scale, repeats, clip_iters, scaled_iters, scales):
    plain = make_input(rows, cols, input_scale)
    reference = torch.softmax(plain, dim=-1)
    encrypted = crypten.cryptensor(plain, src=0)

    results = []
    for name, fn in build_cases(scales, clip_iters, scaled_iters):
        results.append((name, measure(encrypted, reference, fn, repeats)))

    return {
        "rank": comm.get().get_rank(),
        "rows": rows,
        "cols": cols,
        "input_scale": input_scale,
        "repeats": repeats,
        "results": results,
    }


def summarize(outputs):
    meta = outputs[0]
    by_name = {}
    for output in outputs:
        for name, result in output["results"]:
            by_name.setdefault(name, []).append(result)

    print(
        f"world_size=2 shape=({meta['rows']}, {meta['cols']}) "
        f"input_scale={meta['input_scale']} repeats={meta['repeats']}"
    )
    print(
        f"{'case':<20} {'wall_ms':>9} {'comm_ms':>9} {'sys_bytes':>10} "
        f"{'rounds':>7} {'max_abs':>9} {'mean_abs':>9} {'row_sum':>9}"
    )
    for name in sorted(by_name):
        rank_items = by_name[name]
        item = {
            "wall": max(r["wall"] for r in rank_items),
            "comm": max(r["comm"] for r in rank_items),
            "bytes": sum(r["bytes"] for r in rank_items) / 2,
            "rounds": max(r["rounds"] for r in rank_items),
            "max_abs": max(r["max_abs"] for r in rank_items),
            "mean_abs": max(r["mean_abs"] for r in rank_items),
            "row_sum": max(r["row_sum"] for r in rank_items),
        }
        print(
            f"{name:<20} {item['wall'] * 1000:9.3f} {item['comm'] * 1000:9.3f} "
            f"{item['bytes']:10.0f} {item['rounds']:7.1f} {item['max_abs']:9.5f} "
            f"{item['mean_abs']:9.5f} {item['row_sum']:9.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--clip-iters", type=int, default=16)
    parser.add_argument("--scaled-iters", type=int, nargs="+", default=[8, 12, 16, 24])
    parser.add_argument("--scales", type=int, nargs="+", default=[2])
    args = parser.parse_args()

    outputs = run_rank(
        args.rows,
        args.cols,
        args.input_scale,
        args.repeats,
        args.clip_iters,
        sorted(set(args.scaled_iters)),
        args.scales,
    )
    summarize(outputs)


if __name__ == "__main__":
    main()
