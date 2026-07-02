"""
Security analysis of MLFormer inv_sqrt with z_bits=8 (Z ∈ {1,...,255}).

Protocol:
  1) Trusted party (src=0) generates Z, shares [Z] and [Z²]
  2) [W] = [X+eps] × [Z²]   -- 1 Beaver
  3) Reveal W                -- BOTH parties see W
  4) [Y] = [Z] × (1/√W)      -- no comm

Threat: adversary sees W = (X+eps) × Z² and wants to recover X.
"""
import crypten
import torch
import numpy as np


def analyze_candidates():
    """For each (X, Z) pair, count how many Z' give valid X' in BERT range."""
    # BERT LayerNorm variance range
    X_min, X_max = 0.01, 10.0
    Z_max = 255

    x_values = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("=" * 90)
    print("Security Analysis: MLFormer inv_sqrt with z_bits=8, Z ∈ {1,...,255}")
    print(f"Prior: X (variance) ∈ [{X_min}, {X_max}]")
    print("=" * 90)

    print("\n--- Part 1: Candidate count for given W = X × Z² ---")
    print(f"  Adversary sees W, computes candidates X_i = W / i² for i ∈ {{1,...,{Z_max}}}")
    print(f"  Valid candidate: X_i ∈ [{X_min}, {X_max}]")
    print()

    header = f"  {'X':>8} |"
    for z in [1, 3, 5, 10, 50, 100, 200, 255]:
        header += f" Z={z:>3}"
    print(header)
    print(f"  {'':->8}-+-" + "-" * (8 * 8))

    for x in x_values:
        row = f"  {x:>8.3f} |"
        for z in [1, 3, 5, 10, 50, 100, 200, 255]:
            W = x * z * z
            count = 0
            for i in range(1, Z_max + 1):
                xi = W / (i * i)
                if X_min <= xi <= X_max:
                    count += 1
            row += f" {count:>5}"
        print(row)

    print("\n--- Part 2: Expected candidate count over random Z ---")
    print(f"  E[#candidates | X] = avg over Z ∈ {{1,...,{Z_max}}} of valid candidates")
    print()

    print(f"  {'X':>8} {'E[candidates]':>15} {'Min':>6} {'Max':>6} {'P(candidates≤5)':>18}")
    print(f"  {'-'*58}")

    for x in x_values:
        counts = []
        for z in range(1, Z_max + 1):
            W = x * z * z
            count = sum(1 for i in range(1, Z_max + 1)
                        if X_min <= W / (i * i) <= X_max)
            counts.append(count)
        avg = np.mean(counts)
        mn = min(counts)
        mx = max(counts)
        p_le5 = sum(1 for c in counts if c <= 5) / len(counts)
        print(f"  {x:>8.3f} {avg:>15.1f} {mn:>6} {mx:>6} {p_le5:>18.4f}")

    print("\n--- Part 3: Worst-case analysis (unique X recovery) ---")
    print(f"  Cases where adversary can uniquely determine X (candidates=1)")
    print()

    worst_cases = []
    for x in np.arange(X_min, X_max + 0.001, 0.001):
        x = round(x, 3)
        for z in range(1, Z_max + 1):
            W = x * z * z
            count = sum(1 for i in range(1, Z_max + 1)
                        if X_min <= W / (i * i) <= X_max)
            if count == 1:
                worst_cases.append((x, z, W))

    if worst_cases:
        print(f"  Total (X,Z) pairs with unique recovery: {len(worst_cases)}")
        print(f"  Out of total possible pairs: {int((X_max - X_min) / 0.001 + 1) * Z_max}")
        prob = len(worst_cases) / (int((X_max - X_min) / 0.001 + 1) * Z_max)
        print(f"  Probability over random (X,Z): {prob:.6f} ({prob*100:.4f}%)")
        print()
        print(f"  Worst-case X values (first 10):")
        for x, z, W in worst_cases[:10]:
            print(f"    X={x:.3f}, Z={z:>3}, W={W:.6f} → only X={W/z/z:.3f} in range")
    else:
        print("  No unique recovery cases found!")

    print("\n--- Part 4: Comparison across z_bits values ---")
    print(f"  Security vs overflow trade-off")
    print()
    print(f"  {'z_bits':>6} {'Z range':>14} {'cand':>8} "
          f"{'E[cand|X=1]':>12} {'E[cand|X=0.01]':>14} {'overflow':>10}")
    print(f"  {'-'*70}")

    for zb in [2, 4, 6, 8, 10, 12, 14, 16]:
        zmax = (1 << zb) - 1
        total = zmax

        # Compute expected candidates for X=1 and X=0.01
        for xval, label in [(1.0, "X=1"), (0.01, "X=0.01")]:
            counts = []
            for z in range(1, min(zmax + 1, 256)):  # sample up to 255
                W = xval * z * z
                count = sum(1 for i in range(1, min(zmax + 1, 256))
                            if X_min <= W / (i * i) <= X_max)
                counts.append(count)
            if label == "X=1":
                avg1 = np.mean(counts) if counts else 0
            else:
                avg01 = np.mean(counts) if counts else 0

        overflow = zb >= 12  # from our empirical test
        print(f"  {zb:>6} {f'1..{zmax}':>14} {total:>8,} "
              f"{avg1:>12.1f} {avg01:>14.1f} {'YES' if overflow else 'no':>10}")

    print("\n--- Part 5: Party-0 (Z generator) can ALWAYS recover X ---")
    print("  Party 0 generates Z in plaintext, so Z is KNOWN to party 0.")
    print("  After reveal: X = W / Z² - eps.  Party 0 computes this directly.")
    print("  This is independent of z_bits — increasing z_bits does NOT help.")
    print("  Only party 1 (who doesn't know Z) benefits from larger z_bits.")


if __name__ == "__main__":
    analyze_candidates()
