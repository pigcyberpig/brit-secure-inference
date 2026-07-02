#!/bin/bash
######
# Loopback network throttle for local two-party SHAFT/CrypTen runs.
# Supports both the legacy profiles used in this quest and the BLB paper profiles.
######

set -euo pipefail

PROFILE="${1:-}"
DEV="${2:-lo}"

delete_qdisc() {
    sudo tc qdisc del dev "$DEV" root 2>/dev/null || true
}

apply_profile() {
    local rate="$1"
    local one_way_delay="$2"
    delete_qdisc
    sudo tc qdisc add dev "$DEV" root handle 1: tbf rate "$rate" burst 100000 limit 10000
    sudo tc qdisc add dev "$DEV" parent 1:1 handle 10: netem delay "$one_way_delay"
}

case "$PROFILE" in
    ""|help|-h|--help)
        cat <<EOF
Usage: $0 <profile|show|del> [device]

Legacy profiles:
  lan        3 Gbps, 0.5 ms RTT
  lan3g03    3 Gbps, 0.3 ms RTT
  wan4ms     400 Mbps, 4 ms RTT
  wan80ms    100 Mbps, 80 ms RTT

BLB profiles:
  blb-lan    1 Gbps, 0.3 ms RTT
  blb-wan1   400 Mbps, 4 ms RTT
  blb-wan2   100 Mbps, 4 ms RTT
  blb-wan3   100 Mbps, 80 ms RTT

Aliases:
  lan1g      -> blb-lan
  wan400m    -> blb-wan1
  wan100m4   -> blb-wan2
  wan100m80  -> blb-wan3

Utility:
  show       show current qdisc
  del        remove qdisc
EOF
        ;;
    show)
        sudo tc qdisc show dev "$DEV"
        ;;
    del)
        delete_qdisc
        ;;
    lan)
        apply_profile "3000mbit" "0.25msec"
        ;;
    lan3g03)
        apply_profile "3000mbit" "0.15msec"
        ;;
    wan4ms)
        apply_profile "400mbit" "2msec"
        ;;
    wan80ms)
        apply_profile "100mbit" "40msec"
        ;;
    blb-lan|lan1g)
        apply_profile "1000mbit" "0.15msec"
        ;;
    blb-wan1|wan400m)
        apply_profile "400mbit" "2msec"
        ;;
    blb-wan2|wan100m4)
        apply_profile "100mbit" "2msec"
        ;;
    blb-wan3|wan100m80)
        apply_profile "100mbit" "40msec"
        ;;
    *)
        echo "Unknown profile: $PROFILE" >&2
        exit 1
        ;;
esac
