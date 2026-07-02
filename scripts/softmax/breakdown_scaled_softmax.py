import time

import torch

import crypten
import crypten.communicator as comm


def measure(name, fn):
    rank = comm.get().get_rank()
    comm.get().barrier()
    crypten.reset_communication_stats()
    start = time.perf_counter()
    result = fn()
    wall_time = time.perf_counter() - start
    stats = crypten.get_communication_stats()
    if rank == 0:
        print(
            f"{name:<24} wall_ms={wall_time * 1000:8.3f} "
            f"comm_ms={stats['time'] * 1000:8.3f} "
            f"bytes={stats['bytes']:10.0f} rounds={stats['rounds']:5}"
        )
    return result


@crypten.mpc.run_multiprocess(world_size=2)
def run():
    torch.manual_seed(0)
    x_plain = torch.randn(16, 128)
    x = crypten.cryptensor(x_plain, src=0)

    centered = measure("center_mean_sub", lambda: x - x.mean(dim=-1, keepdim=True))
    scaled = measure("divide_public_2", lambda: centered / 2)
    with crypten.cfg.temp_override(
        {"functions.softmax_method": "ode", "functions.softmax_ode_clip": False}
    ):
        p = measure("ode_no_clip_scaled", lambda: scaled.softmax(dim=-1))

    p2 = measure("pow_k2_mul", lambda: p * p)
    total = measure("sum_only", lambda: p2.sum(dim=-1, keepdim=True))
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_total = measure("reciprocal_only", lambda: total.reciprocal())
    measure("final_mul", lambda: p2 * inv_total)

    with crypten.cfg.temp_override(
        {"functions.softmax_method": "ode", "functions.softmax_ode_clip": True}
    ):
        measure("ode_clip_total", lambda: x.softmax(dim=-1))

    with crypten.cfg.temp_override(
        {"functions.softmax_method": "ode", "functions.softmax_ode_clip": False}
    ):
        measure("ode_no_clip_total", lambda: x.softmax(dim=-1))


if __name__ == "__main__":
    run()
