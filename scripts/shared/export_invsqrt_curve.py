"""Export standalone inv_sqrt error curves for NR vs MLFormer.

This script measures the sub-protocol error of 1/sqrt(x + eps) on a synthetic
range without loading BERT or running full 2PC. It is intended for plotting.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import crypten
import torch


EPS = 1e-5
KEY_XS = torch.tensor(
    [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    dtype=torch.float64,
)


class _BeaverCounter:
    def __init__(self):
        self.count = 0
        self._original = None

    def start(self):
        from crypten.mpc.primitives import beaver

        self._original = getattr(beaver, "__beaver_protocol")
        counter = self

        def _counting_wrapper(op, x, y, *args, **kwargs):
            counter.count += 1
            return counter._original(op, x, y, *args, **kwargs)

        setattr(beaver, "__beaver_protocol", _counting_wrapper)

    def stop(self):
        if self._original is not None:
            from crypten.mpc.primitives import beaver

            setattr(beaver, "__beaver_protocol", self._original)


def count_beavers(func):
    counter = _BeaverCounter()
    counter.start()
    t0 = time.time()
    result = func()
    elapsed = time.time() - t0
    count = counter.count
    counter.stop()
    return result, count, elapsed


def resolve_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def inv_sqrt_mlformer(x_enc, eps=EPS, z_bits=8, src=0):
    device = x_enc.device
    size = x_enc.size()
    rank = crypten.communicator.get().get_rank()
    x_shifted = x_enc + eps

    z_plain = torch.zeros(size, device=device, dtype=torch.float64)
    z_sq_plain = torch.zeros(size, device=device, dtype=torch.float64)
    if rank == src:
        max_mag = 1 << z_bits
        mag = torch.randint(
            low=1, high=max_mag, size=size, device=device, dtype=torch.long
        ).to(torch.float64)
        z_plain = mag
        z_sq_plain = mag * mag

    z = crypten.cryptensor(z_plain, src=src, device=device)
    z_sq = crypten.cryptensor(z_sq_plain, src=src, device=device)
    w = (x_shifted * z_sq).get_plain_text().clamp(min=1e-12)
    w_inv_sqrt = 1.0 / torch.sqrt(w)
    return z * w_inv_sqrt


def build_xs(lo, hi, points):
    curve = torch.logspace(math.log10(lo), math.log10(hi), steps=points, dtype=torch.float64)
    mask = (KEY_XS >= lo) & (KEY_XS <= hi)
    xs = torch.cat([curve, KEY_XS[mask]])
    return torch.unique(xs, sorted=True)


def measure_curve(xs, device):
    reference = 1.0 / torch.sqrt(xs + EPS)
    methods = {}
    for name, fn in (
        ("NR", lambda x: x.inv_sqrt()),
        ("MLFormer", lambda x: inv_sqrt_mlformer(x, eps=EPS)),
    ):
        x_enc = crypten.cryptensor(xs.reshape(1, 1, -1).to(device), device=device)
        y, beavers, elapsed = count_beavers(lambda: fn(x_enc))
        y_plain = y.get_plain_text().reshape(-1).cpu()
        abs_err = (y_plain - reference).abs()
        rel_err = abs_err / reference.abs().clamp_min(1e-12)
        methods[name] = {
            "beavers": beavers,
            "elapsed_s": elapsed,
            "max_abs": abs_err.max().item(),
            "mean_abs": abs_err.mean().item(),
            "max_rel": rel_err.max().item(),
            "mean_rel": rel_err.mean().item(),
            "y": y_plain.tolist(),
            "abs_err": abs_err.tolist(),
            "rel_err": rel_err.tolist(),
        }
    return {
        "x": xs.tolist(),
        "reference": reference.tolist(),
        "methods": methods,
    }


def extract_key_points(curve):
    xs = curve["x"]
    key_rows = []
    for key_x in KEY_XS.tolist():
        if key_x < xs[0] or key_x > xs[-1]:
            continue
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - key_x))
        key_rows.append(
            {
                "target_x": key_x,
                "x": xs[idx],
                "NR_abs_err": curve["methods"]["NR"]["abs_err"][idx],
                "NR_rel_err": curve["methods"]["NR"]["rel_err"][idx],
                "MLFormer_abs_err": curve["methods"]["MLFormer"]["abs_err"][idx],
                "MLFormer_rel_err": curve["methods"]["MLFormer"]["rel_err"][idx],
            }
        )
    return key_rows


def write_outputs(json_path, key_csv_path, payload, key_rows):
    json_path = Path(json_path)
    key_csv_path = Path(key_csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    key_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    with key_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_x",
                "x",
                "NR_abs_err",
                "NR_rel_err",
                "MLFormer_abs_err",
                "MLFormer_rel_err",
            ],
        )
        writer.writeheader()
        writer.writerows(key_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=float, default=0.01)
    parser.add_argument("--hi", type=float, default=100.0)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--keypoints-csv-output", required=True)
    args = parser.parse_args()

    crypten.init()
    torch.manual_seed(42)
    device = resolve_device()
    xs = build_xs(args.lo, args.hi, args.points)
    curve = measure_curve(xs, device)
    key_rows = extract_key_points(curve)
    payload = {
        "range": {"lo": args.lo, "hi": args.hi, "points": len(curve["x"]), "eps": EPS},
        "curve": curve,
        "key_points": key_rows,
    }
    write_outputs(args.json_output, args.keypoints_csv_output, payload, key_rows)
    print(
        f"Exported {len(curve['x'])} sweep points and {len(key_rows)} key points to "
        f"{args.json_output} and {args.keypoints_csv_output}"
    )


if __name__ == "__main__":
    main()
