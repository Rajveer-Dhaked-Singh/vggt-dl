# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# Try to import FlashAttention 2.0
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("FlashAttention not found. Falling back to PyTorch scaled_dot_product_attention.")

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
    def forward(self, x: Tensor, pos=None) -> Tensor:
            B, N, C = x.shape
            
            # 1. Project and Reshape to [B, N, 3, H, D]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            
            # Split Q, K, V -> each is [B, N, H, D]
            q, k, v = qkv.unbind(2)

            # Apply Norms
            q, k = self.q_norm(q), self.k_norm(k)

            # Apply RoPE 
            if self.rope is not None:
                # RoPE expects [B, H, N, D], we permute, apply, and permute back
                q = self.rope(q.transpose(1, 2), pos).transpose(1, 2)
                k = self.rope(k.transpose(1, 2), pos).transpose(1, 2)

            # --- OPTIMIZED ATTENTION PATH ---
            
            # Check if we can use Flash Attention 2.0
            # Flash requires half precision and contiguous memory
            if FLASH_AVAILABLE and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
                # ensure_contiguous is vital after the RoPE transpose
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                
                x = flash_attn_func(
                    q, k, v, 
                    dropout_p=self.attn_drop.p if self.training else 0.0, 
                    softmax_scale=self.scale, 
                    causal=False
                )
            
            else:
                # Fallback to PyTorch Native SDPA (which includes its own Flash/MemEff kernels)
                # SDPA expects [B, H, N, D]
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                
                x = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                    scale=self.scale
                )
                x = x.transpose(1, 2) # Back to [B, N, H, D]

            # --- FINAL PROJECTION ---
            x = x.reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
