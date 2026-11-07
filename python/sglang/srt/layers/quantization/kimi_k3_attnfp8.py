"""Quantization config for Kimi-K3-W4AFP8 with FP8 attention projections.

Installed by the checkpoint's apply_patch.py. See that file for why this exists.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

logger = logging.getLogger(__name__)

# Attention projections stored as FP8 in the checkpoint (runtime module names).
FP8_LINEARS = {
    "fused_qkvg_proj",  # KDA layers, fused q/k/v/g (attn_tp == tp)
    "qkv_proj",         # same layers when dp-attention splits the fusion
    "g_proj",           # MLA gate, plus KDA g when unfused
    "o_proj",           # attention output projection
}

# Kept bf16 — these have raw `.weight` readers in kimi_k3.py. Do not add them.
EXCLUDED = ("kv_b_proj", "q_b_proj", "f_b_proj", "f_a_proj", "b_proj")


def _patch_kda_precompile_dtype() -> None:
    """kimi_k3.py uses o_proj.weight.dtype as a stand-in for the activation
    dtype when precompiling the KDA kernel. With o_proj in FP8 that stand-in
    lies and the wrong kernel is compiled — silently wrong, not a crash.
    """
    try:
        from sglang.kernels.ops.attention.fla import kda
        from sglang.srt.models import kimi_k3 as model_mod
    except Exception as exc:  # noqa: BLE001
        logger.warning("kimi_k3_attnfp8: KDA dtype patch skipped (%s)", exc)
        return

    orig = kda.precompile_k3_recompute_w_u_kernel
    if getattr(orig, "_kimi_k3_dtype_fixed", False):
        return

    def wrapped(*, num_heads, dtype, device):
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            dtype = torch.bfloat16
        return orig(num_heads=num_heads, dtype=dtype, device=device)

    wrapped._kimi_k3_dtype_fixed = True
    kda.precompile_k3_recompute_w_u_kernel = wrapped
    if hasattr(model_mod, "precompile_k3_recompute_w_u_kernel"):
        model_mod.precompile_k3_recompute_w_u_kernel = wrapped
    logger.info("kimi_k3_attnfp8: KDA precompile dtype patch applied")


class KimiK3AttnFp8Config(W4AFp8Config):
    """MoE is delegated to W4AFp8; linear routing is decided here."""

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8_attnfp8"

    @classmethod
    def from_config(cls, config):
        self = cls(
            is_checkpoint_fp8_serialized=True,
            is_checkpoint_w4afp8_serialized=True,
            linear_activation_scheme="dynamic",
            moe_activation_scheme="static",
            group_size=int(config.get("group_size", 128)),
        )
        # weight_block_size=None matters: the parent's [128, 128] block quant
        # kills b_proj, and the checkpoint's attention scales are per-tensor.
        self._attn_cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=None,
        )
        _patch_kda_precompile_dtype()
        logger.info("kimi_k3_attnfp8: config ready (FP8 linears: %s)",
                    ", ".join(sorted(FP8_LINEARS)))
        return self

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, LinearBase):
            name = prefix.rsplit(".", 1)[-1] if prefix else ""
            if name in FP8_LINEARS:
                return Fp8LinearMethod(self._attn_cfg)
            # Everything else stays bf16. Sending these to Fp8LinearMethod(self)
            # — what stock does — routes them into the [128,128] block-quant
            # path and kills b_proj (output size 6).
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)
