"""Shared network profiles for BLB-style loopback throttled reruns."""

BLB_NETWORK_PROFILES = {
    "lan_3g_0p3ms": {
        "throttle_arg": "lan3g03",
        "label": "LAN-3G-0.3ms",
        "bandwidth_mbps": 3000,
        "rtt_ms": 0.3,
    },
    "blb_lan": {
        "throttle_arg": "blb-lan",
        "label": "LAN",
        "bandwidth_mbps": 1000,
        "rtt_ms": 0.3,
    },
    "blb_wan1": {
        "throttle_arg": "blb-wan1",
        "label": "WAN1",
        "bandwidth_mbps": 400,
        "rtt_ms": 4.0,
    },
    "blb_wan2": {
        "throttle_arg": "blb-wan2",
        "label": "WAN2",
        "bandwidth_mbps": 100,
        "rtt_ms": 4.0,
    },
    "blb_wan3": {
        "throttle_arg": "blb-wan3",
        "label": "WAN3",
        "bandwidth_mbps": 100,
        "rtt_ms": 80.0,
    },
}
