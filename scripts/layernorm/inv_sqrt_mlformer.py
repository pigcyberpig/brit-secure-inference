"""
MLFormer-style inverse square root protocol for CrypTen LayerNorm.

Protocol:
  Input:  [X] (secret-shared), X > 0 element-wise
  Output: [Y] = 1/sqrt(X)

  1) Trusted party generates positive random Z, shares [Z] and [Z²]
  2) [W] = [X] * [Z²]            — 1 Beaver round
  3) Reveal W
  4) Plaintext: inv_sqrt_W = 1/sqrt(W)
  5) [Y] = [Z] * inv_sqrt_W       — ciphertext * plaintext, no communication

  Verification: Z / sqrt(X·Z²) = Z / (Z·sqrt(X)) = 1/sqrt(X)  (Z > 0)

Compared to default NR inv_sqrt (exp init + 5 NR iters ≈ 23 comm rounds),
this protocol needs only 1 Beaver + 1 reveal.
"""

import math
import time
from unittest.mock import patch

import crypten
import torch
from crypten.config import cfg


# ---------------------------------------------------------------------------
# Beaver round counter (works in in-process mode where comm stats aren't tracked)
# ---------------------------------------------------------------------------
class _BeaverCounter:
    """Monkey-patches Beaver __beaver_protocol to count invocations."""

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


def _count_beavers(func, *args, **kwargs):
    """Run func and return (result, beaver_count, elapsed)."""
    counter = _BeaverCounter()
    counter.start()
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    count = counter.count
    counter.stop()
    return result, count, elapsed


# ---------------------------------------------------------------------------
# MLFormer inv_sqrt protocol
# ---------------------------------------------------------------------------

def _inv_sqrt_mlformer_protocol(
    x,
    *,
    z_bits: int = 8,
    src: int = 0,
    trust_src: bool = True,
    eps: float = 1e-5,
):
    """
    Compute 1/sqrt(x + eps) for secret-shared x using MLFormer-style masking.

    Args:
        x: MPCTensor (element-wise positive expected after +eps)
        z_bits: controls Z range; Z sampled uniformly in [1, 2^z_bits)
        src: rank of trusted party that generates Z
        trust_src: whether to trust src party
        eps: added to x for numerical stability before sqrt

    Returns:
        MPCTensor approximating 1/sqrt(x + eps)
    """
    device = x.device
    size = x.size()
    rank = crypten.communicator.get().get_rank()

    # Shift for numerical stability
    x_shifted = x + eps

    if trust_src:
        # Generate positive random Z and Z² in plaintext
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
    else:
        raise NotImplementedError("Non-trusted-src mode not implemented")

    # [W] = [X+eps] * [Z²] — 1 Beaver round
    W_sh = x_shifted * z_sq

    # Reveal W
    W = W_sh.get_plain_text()

    # Plaintext inverse sqrt
    W = W.clamp(min=1e-12)
    W_inv_sqrt = 1.0 / torch.sqrt(W)

    # [Y] = [Z] * inv_sqrt_W — plaintext × ciphertext, no communication
    Y = z * W_inv_sqrt
    return Y


def inv_sqrt_mlformer(self, eps=1e-5):
    """Wrapper to call MLFormer inv_sqrt from an MPCTensor."""
    return _inv_sqrt_mlformer_protocol(self, eps=eps)


# ---------------------------------------------------------------------------
# LayerNorm with MLFormer inv_sqrt
# ---------------------------------------------------------------------------

def layernorm_mlformer(input_enc, scale_enc, bias_enc, axis=-1, eps=1e-5):
    """
    LayerNorm using MLFormer inv_sqrt instead of default NR inv_sqrt.
    Same computation as crypten.nn.module.LayerNormalization.forward.
    """
    mean = input_enc.mean(dim=axis, keepdim=True)
    variance = input_enc.var(dim=axis, keepdim=True)
    inv_sd = inv_sqrt_mlformer(variance, eps=eps)
    out = (input_enc - mean) * inv_sd
    if scale_enc is not None:
        out = out * scale_enc
    if bias_enc is not None:
        out = out + bias_enc
    return out


def layernorm_original(input_enc, scale_enc, bias_enc, axis=-1, eps=1e-5):
    """LayerNorm using original NR inv_sqrt (no eps in var)."""
    mean = input_enc.mean(dim=axis, keepdim=True)
    inv_sd = input_enc.var(dim=axis, keepdim=True).inv_sqrt()
    out = (input_enc - mean) * inv_sd
    if scale_enc is not None:
        out = out * scale_enc
    if bias_enc is not None:
        out = out + bias_enc
    return out


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark():
    crypten.init()
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, L, H = 1, 128, 768

    # =========================================================================
    # Part 1: standalone inv_sqrt comparison
    # =========================================================================
    print("=" * 90)
    print("Part 1: Standalone inv_sqrt — NR vs MLFormer")
    print("=" * 90)

    sizes_to_test = [(1, 128, 768), (1, 64, 768), (1, 32, 768)]

    for size in sizes_to_test:
        x_plain = torch.rand(size, device=device) * 5.0 + 0.1
        gt = 1.0 / torch.sqrt(x_plain + 1e-5)

        # NR inv_sqrt
        x_enc = crypten.cryptensor(x_plain, device=device)
        y_nr, nr_beavers, t_nr = _count_beavers(x_enc.inv_sqrt)
        nr_plain = y_nr.get_plain_text()

        # MLFormer inv_sqrt
        x_enc = crypten.cryptensor(x_plain, device=device)
        y_mlf, mlf_beavers, t_mlf = _count_beavers(
            lambda: inv_sqrt_mlformer(x_enc, eps=1e-5)
        )
        mlf_plain = y_mlf.get_plain_text()

        nr_mae = (nr_plain - gt).abs().mean().item()
        mlf_mae = (mlf_plain - gt).abs().mean().item()
        nr_max = (nr_plain - gt).abs().max().item()
        mlf_max = (mlf_plain - gt).abs().max().item()

        print(f"\n  Size: {list(size)}")
        print(f"  {'Method':<16} {'MAE':>10} {'Max Err':>10} {'Beavers':>8} {'Time(s)':>10}")
        print(f"  {'-'*58}")
        print(f"  {'NR (5 iters)':<16} {nr_mae:>10.6f} {nr_max:>10.6f} {nr_beavers:>8} {t_nr:>10.4f}")
        print(f"  {'MLFormer':<16} {mlf_mae:>10.6f} {mlf_max:>10.6f} {mlf_beavers:>8} {t_mlf:>10.4f}")

    # Pointwise accuracy across wide range
    print(f"\n  Pointwise accuracy: x in [0.01, 100]")
    test_vals = torch.tensor([0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
                             device=device).reshape(1, 1, -1)
    gt_s = 1.0 / torch.sqrt(test_vals + 1e-5)

    x_enc = crypten.cryptensor(test_vals, device=device)
    nr_s = x_enc.inv_sqrt().get_plain_text()
    x_enc = crypten.cryptensor(test_vals, device=device)
    mlf_s = inv_sqrt_mlformer(x_enc, eps=1e-5).get_plain_text()

    print(f"  {'x':>8} {'1/√x (gt)':>12} {'NR err%':>10} {'MLF err%':>10}")
    print(f"  {'-'*44}")
    for i in range(test_vals.numel()):
        xv = test_vals.flatten()[i].item()
        gv = gt_s.flatten()[i].item()
        nr_err = abs(nr_s.flatten()[i].item() - gv) / abs(gv) * 100
        mlf_err = abs(mlf_s.flatten()[i].item() - gv) / abs(gv) * 100
        print(f"  {xv:>8.2f} {gv:>12.6f} {nr_err:>10.2f}% {mlf_err:>10.2f}%")

    # =========================================================================
    # Part 2: Full LayerNorm comparison
    # =========================================================================
    print("\n" + "=" * 90)
    print("Part 2: Full LayerNorm — Original vs MLFormer inv_sqrt")
    print(f"  Shape: ({B}, {L}, {H}),  hidden_dim={H}")
    print("=" * 90)

    x_plain = torch.randn(B, L, H, device=device)
    scale_plain = torch.randn(H, device=device)
    bias_plain = torch.randn(H, device=device)

    # Ground truth (PyTorch LayerNorm)
    gt_ln = torch.nn.functional.layer_norm(x_plain, [H], scale_plain, bias_plain, eps=1e-5)

    # Original LayerNorm
    x_enc = crypten.cryptensor(x_plain, device=device)
    s_enc = crypten.cryptensor(scale_plain, device=device)
    b_enc = crypten.cryptensor(bias_plain, device=device)

    y_orig, orig_beavers, t_orig = _count_beavers(
        lambda: layernorm_original(x_enc, s_enc, b_enc)
    )
    orig_plain = y_orig.get_plain_text()

    # MLFormer LayerNorm
    x_enc = crypten.cryptensor(x_plain, device=device)
    s_enc = crypten.cryptensor(scale_plain, device=device)
    b_enc = crypten.cryptensor(bias_plain, device=device)

    y_mlf, mlf_beavers, t_mlf = _count_beavers(
        lambda: layernorm_mlformer(x_enc, s_enc, b_enc)
    )
    mlf_plain = y_mlf.get_plain_text()

    # Metrics
    orig_mae = (orig_plain - gt_ln).abs().mean().item()
    orig_max = (orig_plain - gt_ln).abs().max().item()
    mlf_mae = (mlf_plain - gt_ln).abs().mean().item()
    mlf_max = (mlf_plain - gt_ln).abs().max().item()

    # Cosine similarity
    orig_cos = torch.nn.functional.cosine_similarity(
        orig_plain.flatten().unsqueeze(0), gt_ln.flatten().unsqueeze(0)
    ).item()
    mlf_cos = torch.nn.functional.cosine_similarity(
        mlf_plain.flatten().unsqueeze(0), gt_ln.flatten().unsqueeze(0)
    ).item()

    print(f"\n  {'Method':<20} {'MAE':>10} {'Max Err':>10} {'Cos Sim':>10} {'Beavers':>8} {'Time(s)':>10}")
    print(f"  {'-'*72}")
    print(f"  {'Original (NR)':<20} {orig_mae:>10.6f} {orig_max:>10.6f} {orig_cos:>10.6f} {orig_beavers:>8} {t_orig:>10.4f}")
    print(f"  {'MLFormer inv_sqrt':<20} {mlf_mae:>10.6f} {mlf_max:>10.6f} {mlf_cos:>10.6f} {mlf_beavers:>8} {t_mlf:>10.4f}")

    savings = (orig_beavers - mlf_beavers) / orig_beavers * 100
    speedup = t_orig / t_mlf if t_mlf > 0 else float('inf')
    print(f"\n  Beaver reduction: {orig_beavers} → {mlf_beavers}  ({savings:.1f}% fewer)")
    print(f"  Speedup: {speedup:.1f}x")

    # =========================================================================
    # Part 3: Stability test — near-zero variance
    # =========================================================================
    print("\n" + "=" * 90)
    print("Part 3: Stability test — near-zero variance inputs")
    print("=" * 90)

    for sigma in [1.0, 0.1, 0.01, 0.001]:
        x_small = torch.randn(B, 8, H, device=device) * sigma
        gt_small = torch.nn.functional.layer_norm(x_small, [H], scale_plain, bias_plain, eps=1e-5)

        x_enc = crypten.cryptensor(x_small, device=device)
        s_enc = crypten.cryptensor(scale_plain, device=device)
        b_enc = crypten.cryptensor(bias_plain, device=device)

        y_orig = layernorm_original(x_enc, s_enc, b_enc).get_plain_text()
        x_enc = crypten.cryptensor(x_small, device=device)
        s_enc = crypten.cryptensor(scale_plain, device=device)
        b_enc = crypten.cryptensor(bias_plain, device=device)
        y_mlf = layernorm_mlformer(x_enc, s_enc, b_enc).get_plain_text()

        orig_mae = (y_orig - gt_small).abs().mean().item()
        mlf_mae = (y_mlf - gt_small).abs().mean().item()

        # Check for NaN/Inf
        orig_has_nan = y_orig.isnan().any().item()
        mlf_has_nan = y_mlf.isnan().any().item()

        print(f"  σ={sigma:<8} Original MAE={orig_mae:.6f}  MLFormer MAE={mlf_mae:.6f}  "
              f"NaN: orig={orig_has_nan} mlf={mlf_has_nan}")


if __name__ == "__main__":
    benchmark()
