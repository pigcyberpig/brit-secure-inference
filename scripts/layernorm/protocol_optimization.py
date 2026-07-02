"""
协议优化分析：能否解决安全性和溢出问题？

问题 A: Party 0 知道 Z → 恢复 X       → 解决方案: 协作生成 Z
问题 B: Z 太小 → 无法完美掩码          → 解决方案: 分步乘法避免溢出
"""
import crypten
import torch
import time
import numpy as np


def inv_sqrt_collaborative(x_enc, z_bits_each=6, eps=1e-5):
    """
    优化协议: 双方协作生成 Z, 任何一方都不知道 Z.

    步骤:
      1) Party 0 生成 [A]  (A ∈ {1,...,2^k-1})
      2) Party 1 生成 [B]  (B ∈ {1,...,2^k-1})
      3) [Z] = [A] × [B]           ← Beaver #1
      4) [Z²] = [Z] × [Z]          ← Beaver #2
      5) [W] = [X+eps] × [Z²]      ← Beaver #3
      6) Reveal W = X × Z²
      7) [Y] = [Z] × (1/√W)        ← local

    Z = A×B ∈ {1,...,(2^k-1)²}
    任何一方只知道自己的分量, 不知道 Z.
    """
    device = x_enc.device
    size = x_enc.size()
    rank = crypten.communicator.get().get_rank()

    x_shifted = x_enc + eps

    # Step 1-2: Generate two independent random components
    # In single-process mode both come from rank 0; in 2PC each party
    # would generate its own. The Beaver multiplications are the same.
    max_mag = 1 << z_bits_each
    a_plain = torch.randint(low=1, high=max_mag, size=size,
                            device=device, dtype=torch.long).to(torch.float64)
    b_plain = torch.randint(low=1, high=max_mag, size=size,
                            device=device, dtype=torch.long).to(torch.float64)

    a_enc = crypten.cryptensor(a_plain, src=0, device=device)
    b_enc = crypten.cryptensor(b_plain, src=0, device=device)

    # Step 3: [Z] = [A] × [B]  — Beaver #1
    z = a_enc * b_enc

    # Step 4: [Z²] = [Z] × [Z]  — Beaver #2
    z_sq = z * z

    # Step 5: [W] = [X+eps] × [Z²]  — Beaver #3
    W = (x_shifted * z_sq).get_plain_text()

    # Step 6: Plaintext inverse sqrt
    W = W.clamp(min=1e-12)
    W_inv_sqrt = 1.0 / torch.sqrt(W)

    # Step 7: [Y] = [Z] × (1/√W) — local
    return z * W_inv_sqrt


def inv_sqrt_original_nr(x_enc, eps=1e-5):
    """Original NR inv_sqrt."""
    return x_enc.inv_sqrt()


class _BeaverCounter:
    def __init__(self):
        self.count = 0
        self._original = None

    def start(self):
        from crypten.mpc.primitives import beaver
        self._original = getattr(beaver, '__beaver_protocol')
        counter = self
        def _counting_wrapper(op, x, y, *args, **kwargs):
            counter.count += 1
            return counter._original(op, x, y, *args, **kwargs)
        setattr(beaver, '__beaver_protocol', _counting_wrapper)

    def stop(self):
        if self._original is not None:
            from crypten.mpc.primitives import beaver
            setattr(beaver, '__beaver_protocol', self._original)


def count_beavers(func, *args, **kwargs):
    counter = _BeaverCounter()
    counter.start()
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    count = counter.count
    counter.stop()
    return result, count, elapsed


def main():
    crypten.init()
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("协议优化: 双方协作生成 Z")
    print("=" * 80)

    # =========================================================================
    # Part 1: Overflow analysis for different z_bits_each
    # =========================================================================
    print("\n--- Part 1: 溢出分析 ---")
    print("  约束: X × Z₀² × Z₁² × 2³² < 2⁶³ (Beaver #3 中间步骤)")
    print("  即: X × Z_max⁴ < 2³¹")
    print()

    print(f"  {'z_bits':>6} {'Z_i range':>12} {'Z=Z₀×Z₁ max':>14} {'Z² max':>16} {'X×Z⁴_max':>14} {'2³¹?':>6}")
    print(f"  {'-'*72}")

    for zb in [4, 5, 6, 7, 8]:
        zi_max = (1 << zb) - 1
        z_max = zi_max * zi_max
        zsq_max = z_max * z_max
        product = 10.0 * zsq_max  # X_max=10
        safe = product < 2**31
        print(f"  {zb:>6} {f'1..{zi_max}':>12} {z_max:>14,} {zsq_max:>16,} {product:>14,.0f} {'✓' if safe else '✗':>6}")

    # =========================================================================
    # Part 2: Benchmark different z_bits_each
    # =========================================================================
    print("\n--- Part 2: 精度与效率测试 (X ∈ BERT 方差范围) ---")
    print()

    x_vals = torch.tensor([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                           device=device, dtype=torch.float64).reshape(1, 1, -1)
    gt = 1.0 / torch.sqrt(x_vals + 1e-5)

    # NR baseline
    x_enc = crypten.cryptensor(x_vals, device=device)
    y_nr, nr_beavers, t_nr = count_beavers(inv_sqrt_original_nr, x_enc)
    nr_plain = y_nr.get_plain_text()
    nr_err = ((nr_plain - gt) / gt).abs().max().item() * 100

    print(f"  NR baseline: {nr_beavers} Beavers, max err {nr_err:.4f}%, time {t_nr*1000:.1f}ms")
    print()

    print(f"  {'z_bits':>6} {'Z range':>14} {'Beavers':>8} {'max_err%':>10} {'time_ms':>10} {'status':>8}")
    print(f"  {'-'*62}")

    for zb in [4, 5, 6, 7, 8]:
        x_enc = crypten.cryptensor(x_vals, device=device)
        try:
            y, bv, t = count_beavers(inv_sqrt_collaborative, x_enc, z_bits_each=zb)
            y_plain = y.get_plain_text()
            err = ((y_plain - gt) / gt).abs().max().item() * 100
            zi = (1 << zb) - 1
            z = zi * zi
            status = "OK" if err < 5 else "FAIL"
            print(f"  {zb:>6} {f'1..{z}':>14} {bv:>8} {err:>10.4f} {t*1000:>10.1f} {status:>8}")
        except Exception as e:
            print(f"  {zb:>6} {'—':>14} {'—':>8} {'CRASH':>10}")

    # =========================================================================
    # Part 3: Security comparison
    # =========================================================================
    print("\n--- Part 3: 安全性对比 ---")
    print()
    print("  ┌───────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ 方法                  │ Beavers  │ Party 0  │ Party 1  │ 可证明?  │")
    print("  ├───────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print("  │ NR (原始)             │   12     │ 不知道X  │ 不知道X  │ ✓        │")
    print("  │ MLFormer z_bits=2     │    1     │ 知道X!   │ 3候选    │ ✗        │")
    print("  │ MLFormer z_bits=8     │    1     │ 知道X!   │ 255候选  │ ✗        │")
    print("  │ 协作Z (z_bits=6)      │    3     │ 63候选   │ 63候选   │ ✗        │")
    print("  │ 协作Z (z_bits=7)      │    3     │ 127候选  │ 127候选  │ ✗        │")
    print("  │ 理想 (Z ∈ 𝔽_p)       │    1     │ 不知道X  │ 不知道X  │ ✓        │")
    print("  └───────────────────────┴──────────┴──────────┴──────────┴──────────┘")

    # =========================================================================
    # Part 4: Why provable security is fundamentally impossible in CrypTen
    # =========================================================================
    print("""
--- Part 4: 为什么在 CrypTen 中无法实现可证明安全 ---
  根本原因: CrypTen 使用定点数 (int64 + precision_bits=16), 不是素数域 𝔽_p

  在素数域 𝔽_p (p ≈ 2⁶⁴) 中:
    W = X × Z² mod p, Z 均匀于 𝔽_p*
    → W 均匀于 𝔽_p* (完美掩码)
    → 模拟器可以生成均匀 W' → 可证明安全 ✓

  在定点数 Z_{2⁶⁴} 中:
    W = X × Z², Z ∈ {1,...,k} (k << 2⁶⁴)
    → W 只取 k 个离散值 (不均匀)
    → 模拟器无法生成正确分布的 W' → 不可证明 ✗

  即使不考虑溢出 (假设用 int128):
    precision_bits=16 意味着 X 编码为 X × 2^16
    Z 编码为 Z × 2^16, Z 最大 ≈ 2^31 (保持乘积在 int128 内)
    2^31 ≈ 2 × 10⁹ 个候选值 — 看似很多, 但远少于 𝔽_p 的 2^64 个

  结论: 只要用定点数, Z 就不可能覆盖整个域, 掩码就不完美
       → 理想/现实安全性在 CrypTen 框架下无法证明
""")


if __name__ == "__main__":
    main()
