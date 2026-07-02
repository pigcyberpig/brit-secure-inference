"""
Comprehensive benchmark: NR vs MLFormer z8 vs MLFormer fractional-Z (optimal).
Compares accuracy, beaver triples, timing, and LayerNorm end-to-end.
"""
import crypten
import torch
import time
import numpy as np


class _BeaverCounter:
    def __init__(self):
        self.count = 0
        self._original = None

    def start(self):
        from crypten.mpc.primitives import beaver
        self._original = getattr(beaver, '__beaver_protocol')
        counter = self
        def _w(op, x, y, *a, **k):
            counter.count += 1
            return counter._original(op, x, y, *a, **k)
        setattr(beaver, '__beaver_protocol', _w)

    def stop(self):
        if self._original:
            from crypten.mpc.primitives import beaver
            setattr(beaver, '__beaver_protocol', self._original)


def cb(func, *args, **kwargs):
    c = _BeaverCounter()
    c.start()
    t0 = time.time()
    r = func(*args, **kwargs)
    e = time.time() - t0
    n = c.count
    c.stop()
    return r, n, e


# ---------------------------------------------------------------------------
# Three inv_sqrt methods
# ---------------------------------------------------------------------------

def inv_sqrt_nr(x_enc, eps=1e-5):
    """NR baseline (CrypTen default)."""
    return x_enc.inv_sqrt()


def inv_sqrt_z8(x_enc, eps=1e-5):
    """MLFormer with integer Z, z_bits=8 (255 candidates)."""
    dev = x_enc.device
    sz = x_enc.size()
    rk = crypten.communicator.get().get_rank()
    xs = x_enc + eps
    zp = torch.zeros(sz, device=dev, dtype=torch.float64)
    zsp = torch.zeros(sz, device=dev, dtype=torch.float64)
    if rk == 0:
        mag = torch.randint(1, 256, sz, device=dev, dtype=torch.long).to(torch.float64)
        zp = mag
        zsp = mag * mag
    z = crypten.cryptensor(zp, src=0, device=dev)
    zs = crypten.cryptensor(zsp, src=0, device=dev)
    W = (xs * zs).get_plain_text().clamp(min=1e-12)
    return z * (1.0 / torch.sqrt(W))


def inv_sqrt_frac(x_enc, max_k=5_000_000, scale=50.0, eps=1e-5):
    """MLFormer with fractional Z = k/2^16 (optimal config)."""
    dev = x_enc.device
    sz = x_enc.size()
    rk = crypten.communicator.get().get_rank()
    xs = x_enc / scale + eps
    zp = torch.zeros(sz, device=dev, dtype=torch.float64)
    zsp = torch.zeros(sz, device=dev, dtype=torch.float64)
    if rk == 0:
        k = torch.randint(1, max_k + 1, sz, device=dev, dtype=torch.long)
        zp = k.to(torch.float64) / (2**16)
        zsp = zp * zp
    z = crypten.cryptensor(zp, src=0, device=dev)
    zs = crypten.cryptensor(zsp, src=0, device=dev)
    W = (xs * zs).get_plain_text().clamp(min=1e-12)
    return z * (1.0 / torch.sqrt(W)) * (1.0 / np.sqrt(scale))


# ---------------------------------------------------------------------------
# LayerNorm wrappers
# ---------------------------------------------------------------------------

def layernorm(x, s, b, inv_sqrt_fn, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    inv_sd = inv_sqrt_fn(var, eps=eps)
    out = (x - mean) * inv_sd
    if s is not None:
        out = out * s
    if b is not None:
        out = out + b
    return out


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def main():
    crypten.init()
    torch.manual_seed(42)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    methods = [
        ("NR (12 Beavers)",    inv_sqrt_nr),
        ("MLF z8 (255 cand)",  inv_sqrt_z8),
        ("MLF frac (5M cand)", inv_sqrt_frac),
    ]

    sep = "=" * 110

    # =========================================================================
    # Part 1: Pointwise inv_sqrt accuracy
    # =========================================================================
    print(sep)
    print("Part 1: inv_sqrt 逐点精度  (X in BERT variance range)")
    print(sep)

    x_vals = torch.tensor([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                           device=dev, dtype=torch.float64).reshape(1, 1, -1)
    gt = 1.0 / torch.sqrt(x_vals + 1e-5)

    results_inv = {}
    for name, fn in methods:
        x_enc = crypten.cryptensor(x_vals, device=dev)
        y, bv, t = cb(fn, x_enc)
        yp = y.get_plain_text()
        errs = ((yp - gt) / gt).abs() * 100
        results_inv[name] = (yp, errs, bv, t)

    hdr = f"{'X':>8}"
    for name, _, _, _ in [('NR', 0, 0, 0), ('MLF z8', 0, 0, 0), ('MLF frac', 0, 0, 0)]:
        hdr += f" {name + ' err%':>14}"
    print(hdr)
    print("-" * len(hdr))

    x_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    names_short = ["NR (12 Beavers)", "MLF z8 (255 cand)", "MLF frac (5M cand)"]
    for i, xv in enumerate(x_list):
        line = f"{xv:>8.3f}"
        for name in names_short:
            _, errs, _, _ = results_inv[name]
            line += f" {errs.flatten()[i].item():>13.4f}%"
        print(line)

    print(f"\n  {'Method':<24} {'Beavers':>8} {'Max err%':>10} {'Time(ms)':>10} {'Candidates':>12}")
    print(f"  {'-'*68}")
    for name, fn in methods:
        yp, errs, bv, t = results_inv[name]
        mx = errs.max().item()
        cand = {"NR": "∞", "MLF z8": "255", "MLF frac": "5,000,000"}
        c = [v for k, v in cand.items() if k in name][0]
        print(f"  {name:<24} {bv:>8} {mx:>10.4f} {t*1000:>10.1f} {c:>12}")

    # =========================================================================
    # Part 2: Full LayerNorm end-to-end (1, 128, 768)
    # =========================================================================
    print(f"\n{sep}")
    print("Part 2: 完整 LayerNorm 端到端  (B=1, L=128, H=768)")
    print(sep)

    B, L, H = 1, 128, 768
    x_plain = torch.randn(B, L, H, device=dev)
    sc_p = torch.randn(H, device=dev)
    bi_p = torch.randn(H, device=dev)
    gt_ln = torch.nn.functional.layer_norm(x_plain, [H], sc_p, bi_p, eps=1e-5)

    ln_methods = [
        ("NR",                lambda x, s, b: layernorm(x, s, b, inv_sqrt_nr)),
        ("MLF z8",            lambda x, s, b: layernorm(x, s, b, inv_sqrt_z8)),
        ("MLF frac (5M S50)", lambda x, s, b: layernorm(x, s, b, inv_sqrt_frac)),
    ]

    print(f"\n  {'Method':<24} {'MAE':>12} {'Max Err':>12} {'CosSim':>10} {'Beavers':>8} {'Time(ms)':>10}")
    print(f"  {'-'*80}")

    ln_results = {}
    for name, ln_fn in ln_methods:
        x_enc = crypten.cryptensor(x_plain, device=dev)
        s_enc = crypten.cryptensor(sc_p, device=dev)
        b_enc = crypten.cryptensor(bi_p, device=dev)
        y, bv, t = cb(lambda: ln_fn(x_enc, s_enc, b_enc))
        yp = y.get_plain_text()
        mae = (yp - gt_ln).abs().mean().item()
        mx = (yp - gt_ln).abs().max().item()
        cs = cos_sim(yp, gt_ln)
        ln_results[name] = (mae, mx, cs, bv, t)
        print(f"  {name:<24} {mae:>12.6f} {mx:>12.6f} {cs:>10.6f} {bv:>8} {t*1000:>10.1f}")

    # =========================================================================
    # Part 3: Multiple LayerNorm shapes
    # =========================================================================
    print(f"\n{sep}")
    print("Part 3: 不同 LayerNorm 形状对比")
    print(sep)

    shapes = [(1, 32, 768), (1, 64, 768), (1, 128, 768), (2, 128, 768)]
    print(f"\n  {'Shape':<16} {'Method':<20} {'MAE':>12} {'Beavers':>8} {'Time(ms)':>10}")
    print(f"  {'-'*70}")

    for shape in shapes:
        B_s, L_s, H_s = shape
        xp = torch.randn(*shape, device=dev)
        sp = torch.randn(H_s, device=dev)
        bp = torch.randn(H_s, device=dev)
        gt_s = torch.nn.functional.layer_norm(xp, [H_s], sp, bp, eps=1e-5)

        for name, ln_fn in ln_methods:
            x_enc = crypten.cryptensor(xp, device=dev)
            s_enc = crypten.cryptensor(sp, device=dev)
            b_enc = crypten.cryptensor(bp, device=dev)
            y, bv, t = cb(lambda: ln_fn(x_enc, s_enc, b_enc))
            yp = y.get_plain_text()
            mae = (yp - gt_s).abs().mean().item()
            print(f"  {str(shape):<16} {name:<20} {mae:>12.6f} {bv:>8} {t*1000:>10.1f}")

    # =========================================================================
    # Part 4: Stability near zero
    # =========================================================================
    print(f"\n{sep}")
    print("Part 4: 稳定性测试 — 小方差输入")
    print(sep)

    print(f"\n  {'sigma':<10} {'Method':<20} {'MAE':>12} {'Has NaN':>10}")
    print(f"  {'-'*56}")

    for sigma in [1.0, 0.1, 0.01, 0.001]:
        xs = torch.randn(1, 32, 768, device=dev) * sigma
        gt_s = torch.nn.functional.layer_norm(xs, [768], sc_p, bi_p, eps=1e-5)
        for name, ln_fn in ln_methods:
            x_enc = crypten.cryptensor(xs, device=dev)
            s_enc = crypten.cryptensor(sc_p, device=dev)
            b_enc = crypten.cryptensor(bi_p, device=dev)
            yp = ln_fn(x_enc, s_enc, b_enc).get_plain_text()
            mae = (yp - gt_s).abs().mean().item()
            has_nan = yp.isnan().any().item()
            print(f"  {sigma:<10.3f} {name:<20} {mae:>12.6f} {str(has_nan):>10}")

    # =========================================================================
    # Part 5: Summary
    # =========================================================================
    nr_mae = ln_results["NR"][0]
    z8_mae = ln_results["MLF z8"][0]
    fr_mae = ln_results["MLF frac (5M S50)"][0]
    nr_bv = ln_results["NR"][3]
    z8_bv = ln_results["MLF z8"][3]
    fr_bv = ln_results["MLF frac (5M S50)"][3]
    nr_t = ln_results["NR"][4]
    z8_t = ln_results["MLF z8"][4]
    fr_t = ln_results["MLF frac (5M S50)"][4]

    print(f"""
{sep}
Part 5: 总结
{sep}

  ┌───────────────────┬──────────┬──────────┬───────────┬──────────┬────────────┐
  │ Method            │ Beavers  │ MAE      │ CosSim    │ Time(ms) │ Candidates │
  ├───────────────────┼──────────┼──────────┼───────────┼──────────┼────────────┤
  │ NR (baseline)     │ {nr_bv:>8} │ {nr_mae:>8.6f} │ {ln_results['NR'][2]:>9.6f} │ {nr_t*1000:>8.1f} │          ∞ │
  │ MLF z_bits=8      │ {z8_bv:>8} │ {z8_mae:>8.6f} │ {ln_results['MLF z8'][2]:>9.6f} │ {z8_t*1000:>8.1f} │        255 │
  │ MLF frac 5M S50   │ {fr_bv:>8} │ {fr_mae:>8.6f} │ {ln_results['MLF frac (5M S50)'][2]:>9.6f} │ {fr_t*1000:>8.1f} │  5,000,000 │
  └───────────────────┴──────────┴──────────┴───────────┴──────────┴────────────┘

  Beaver 节省: NR {nr_bv} → MLF {z8_bv}  ({(nr_bv-z8_bv)/nr_bv*100:.0f}% 减少)
  候选空间: z8 255 → frac 5,000,000  (19,600× 扩大)
  安全性: frac 方案 Party 1 面对约 {int(5000000*0.97):,} 个有效候选，猜测概率 ≤ 2×10⁻⁷
""")


if __name__ == "__main__":
    main()
