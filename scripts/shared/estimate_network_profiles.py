#!/usr/bin/env python3
"""Estimate paper-style communication time under fixed network profiles."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict


NETWORK_PROFILES = OrderedDict(
    [
        (
            "lan_3g_0p5ms",
            {
                "bandwidth_Bps": 3_000_000_000 / 8,
                "latency_s": 0.0005,
                "short_name": "LAN",
                "label": "3 Gbps, 0.5 ms",
                "reason": "Representative LAN profile after correcting SHAFT's byte/bit bandwidth mismatch.",
            },
        ),
        (
            "wan_400m_4ms",
            {
                "bandwidth_Bps": 400_000_000 / 8,
                "latency_s": 0.004,
                "short_name": "WAN-4ms",
                "label": "400 Mbps, 4 ms",
                "reason": "Representative moderate-WAN profile aligned with recent secure Transformer inference work.",
            },
        ),
        (
            "wan_100m_80ms",
            {
                "bandwidth_Bps": 100_000_000 / 8,
                "latency_s": 0.080,
                "short_name": "WAN-80ms",
                "label": "100 Mbps, 80 ms",
                "reason": "Representative high-latency WAN profile used to expose round-sensitive speedups.",
            },
        ),
    ]
)


def estimate_time_s(compute_time_s: float, comm_bytes: int, rounds: int, profile: dict) -> float:
    return (
        float(compute_time_s)
        + 2.0 * int(comm_bytes) / float(profile["bandwidth_Bps"])
        + int(rounds) * float(profile["latency_s"])
    )


def estimate_all(compute_time_s: float, comm_bytes: int, rounds: int) -> OrderedDict:
    estimates = OrderedDict()
    for name, profile in NETWORK_PROFILES.items():
        estimates[name] = {
            "estimated_time_s": estimate_time_s(compute_time_s, comm_bytes, rounds, profile),
            **profile,
        }
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-time-s", type=float, required=True)
    parser.add_argument("--comm-bytes", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate_all(args.compute_time_s, args.comm_bytes, args.rounds), indent=2))


if __name__ == "__main__":
    main()
