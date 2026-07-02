#!/usr/bin/env python3
"""Public-mask-aware softmax primitives and a BERT attention adapter.

This module is intentionally a small integration shim. It does not rewrite ONNX
graphs and does not modify CrypTen internals; callers install the adapter on a
converted private BERT model and set the public attention mask before forward.
"""

from __future__ import annotations

from types import MethodType

import crypten as ct
import torch
from crypten.config import cfg
from crypten.cryptensor import CrypTensor


def parse_scaled_method(name: str) -> tuple[int, int]:
    scale_part, iter_part = name[len("scaled_k") :].split("_i", 1)
    return int(scale_part), int(iter_part)


def require_public_mask(mask: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, CrypTensor):
        raise TypeError("public-mask-aware softmax requires a public torch.Tensor mask")
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    return mask


def attention_key_mask(mask: torch.Tensor, input_tensor: CrypTensor) -> torch.Tensor:
    """Broadcast a public [B, K] attention mask to a [B, 1, 1, K] softmax mask."""
    public = require_public_mask(mask).to(device=input_tensor.device, dtype=torch.float32)
    while public.dim() < input_tensor.dim():
        public = public.unsqueeze(1)
    return public


def masked_sum(x: CrypTensor, mask: torch.Tensor, dim: int, keepdim: bool = True) -> CrypTensor:
    """Sum only valid entries. `mask` must already broadcast to `x`."""
    return (x * mask).sum(dim=dim, keepdim=keepdim)


def masked_mean(x: CrypTensor, mask: torch.Tensor, dim: int, keepdim: bool = True) -> CrypTensor:
    """Mean only valid entries. `mask` must already broadcast to `x`."""
    total = masked_sum(x, mask, dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return total / count


def masked_reciprocal_softmax(x: CrypTensor, mask: torch.Tensor, dim: int) -> CrypTensor:
    masked_x = x * mask + (1.0 - mask) * -10000.0
    maximum = masked_x.max(dim, keepdim=True)[0]
    numerator = (x - maximum).exp() * mask
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_denominator = numerator.sum(dim=dim, keepdim=True).reciprocal()
    return numerator * inv_denominator * mask


def masked_ode_softmax(
    x: CrypTensor,
    mask: torch.Tensor,
    dim: int,
    iter_num: int,
    clip: bool,
    lower: float,
    upper: float,
) -> CrypTensor:
    if clip:
        diff = ct.cat([x - upper, lower - x]).relu().split(x.shape[0])
        x = x + diff[1] - diff[0]

    x = x / iter_num
    count = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    g = x.new(mask.expand(tuple(x.shape)) / count, device=x.device)

    for _ in range(iter_num):
        gx_sum = masked_sum(g * x, mask, dim=dim, keepdim=True)
        g = g + ((x - gx_sum) * g)
        g = g * mask
    return g


def masked_scaled_softmax(x: CrypTensor, mask: torch.Tensor, dim: int, method: str) -> CrypTensor:
    scale, iter_num = parse_scaled_method(method)
    centered = x - masked_mean(x, mask, dim=dim, keepdim=True)
    scaled = centered / scale
    probs = masked_ode_softmax(
        scaled,
        mask,
        dim=dim,
        iter_num=iter_num,
        clip=False,
        lower=cfg.functions.softmax_ode_lb,
        upper=cfg.functions.softmax_ode_ub,
    )

    powered = probs
    if scale > 1:
        exponent = scale
        result = None
        base = probs
        while exponent:
            if exponent & 1:
                result = base if result is None else result * base
            exponent >>= 1
            if exponent:
                base = base * base
        powered = result

    denom = masked_sum(powered, mask, dim=dim, keepdim=True)
    with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inv_total = denom.reciprocal()
    return powered * inv_total * mask


def masked_softmax(
    x: CrypTensor,
    mask: torch.Tensor,
    dim: int,
    *,
    mask_layout: str = "broadcast",
) -> CrypTensor:
    """Dispatch to the current CrypTen softmax backend with public mask semantics."""
    if mask_layout == "attention_key":
        public_mask = attention_key_mask(mask, x)
    elif mask_layout == "broadcast":
        public_mask = require_public_mask(mask).to(device=x.device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown mask_layout: {mask_layout}")

    method = cfg.functions.softmax_method
    if method == "ode":
        return masked_ode_softmax(
            x,
            public_mask,
            dim=dim,
            iter_num=cfg.functions.softmax_ode_iter_num,
            clip=cfg.functions.softmax_ode_clip,
            lower=cfg.functions.softmax_ode_lb,
            upper=cfg.functions.softmax_ode_ub,
        )
    if method.startswith("scaled_k") and "_i" in method:
        return masked_scaled_softmax(x, public_mask, dim=dim, method=method)
    if method in {"ideal", "reciprocal"}:
        return masked_reciprocal_softmax(x, public_mask, dim=dim)
    raise ValueError(f"masked softmax does not support method {method}")


class BertPublicMaskedAttentionAdapter:
    """Install public-mask-aware softmax on CrypTen-converted BERT attention nodes."""

    def __init__(self, private_model, node_pattern: str = "/attention/self/Softmax_output_0"):
        self.private_model = private_model
        self.node_pattern = node_pattern
        self.current_attention_mask = None
        self.original_forwards = {}

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        self.current_attention_mask = require_public_mask(attention_mask).detach()

    def install(self) -> int:
        if self.original_forwards:
            return len(self.original_forwards)

        for name, module in self.private_model._modules.items():
            if self.node_pattern not in name:
                continue
            original = object.__getattribute__(module, "forward")
            self.original_forwards[name] = original

            def wrapped(module_self, input_tensor, _orig=original):
                if self.current_attention_mask is None:
                    return _orig(input_tensor)
                return masked_softmax(
                    input_tensor,
                    self.current_attention_mask,
                    module_self.dim,
                    mask_layout="attention_key",
                )

            module.forward = MethodType(wrapped, module)
        return len(self.original_forwards)

    def uninstall(self) -> None:
        for name, original in self.original_forwards.items():
            self.private_model._modules[name].forward = original
        self.original_forwards.clear()
        self.current_attention_mask = None


def install_public_masked_bert_attention(private_model) -> BertPublicMaskedAttentionAdapter:
    adapter = BertPublicMaskedAttentionAdapter(private_model)
    installed = adapter.install()
    if installed == 0:
        raise RuntimeError("no BERT attention Softmax nodes matched the adapter pattern")
    return adapter
