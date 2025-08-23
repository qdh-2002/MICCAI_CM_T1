import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


# -------------------------------
# Utilities
# -------------------------------
def compute_psf_from_mask(mask):
    """
    Compute the Point Spread Function (PSF) from a k-space sampling mask.
    
    Args:
        mask: [B, 1, H, W] binary sampling mask in k-space
    
    Returns:
        psf: [B, 1, H, W] PSF in image domain with unit sum normalization
    """
    mask_complex = mask.to(th.complex64)
    psf_complex = th.fft.ifft2(th.fft.ifftshift(mask_complex, dim=(-2, -1)), norm='ortho')
    psf_complex = th.fft.fftshift(psf_complex, dim=(-2, -1))
    psf = th.abs(psf_complex)
    psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return psf


class PromptEncoder(nn.Module):
    """
    Encodes a PSF (Point Spread Function) into spatial embeddings to inject into attention layers.
    """
    def __init__(self, in_channels, embed_dim, use_conv=True, dims=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_conv = use_conv

        if use_conv:
            self.encoder = nn.Sequential(
                conv_nd(dims, in_channels, embed_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                conv_nd(dims, embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        else:
            # global vector encoder; not used here
            self.encoder = linear(in_channels, embed_dim)

    def forward(self, psf):
        """
        :param psf: [B, 1, H, W] Point Spread Function
        :return: prompt features [B, embed_dim, H, W]
        """
        return self.encoder(psf)


# -------------------------------
# Core Blocks
# -------------------------------
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # [B, C, HW]
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # [B, C, HW+1]
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)  # <-- fixed bug
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential that passes timestep embeddings to children that support it.
    Also forwards an optional `prompt` to any AttentionBlock in the sequence.
    """
    def forward(self, x, emb, prompt=None):
        for layer in self:
            # ResBlocks or any TimestepBlock
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            # AttentionBlocks get the prompt feature map via `mask=...`
            elif isinstance(layer, AttentionBlock):
                x = layer(x, mask=prompt)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling with optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling with optional convolution.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block that can optionally change the number of channels.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Spatial self-attention with optional cross-attention to external prompt features (PSF).
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="flash",
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
        prompt_channels=None,  # channels of the prompt (PSF) feature map
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)

        # Cross-attention projections for PSF conditioning
        prompt_input_channels = prompt_channels if prompt_channels is not None else channels
        self.k_cross_proj = conv_nd(dims, prompt_input_channels, channels, 1)
        self.v_cross_proj = conv_nd(dims, prompt_input_channels, channels, 1)

        self.attention_type = attention_type
        if attention_type == "flash":
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.use_attention_checkpoint = not (self.use_checkpoint or self.attention_type == "flash")
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x, mask=None):
        if mask is not None:
            return checkpoint(self._forward, (x, mask), self.parameters(), self.use_checkpoint)
        else:
            return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x, mask=None):
        b, c, *spatial = x.shape
        T = int(np.prod(spatial))
        qkv = self.qkv(self.norm(x)).view(b, -1, T)

        if mask is not None:
            # Resize prompt to current spatial size
            mask_resized = F.interpolate(mask, size=spatial, mode="bilinear", align_corners=False)
            mask_resized = mask_resized.to(x.dtype).to(x.device)

            # Cross-attention K/V from prompt
            k_cross = self.k_cross_proj(mask_resized).view(b, -1, T)
            v_cross = self.v_cross_proj(mask_resized).view(b, -1, T)

            # Split original qkv
            q, k, v = qkv.chunk(3, dim=1)

            # Concatenate along sequence dimension (self+cross attention)
            k_extended = th.cat([k, k_cross], dim=2)
            v_extended = th.cat([v, v_cross], dim=2)

            if self.attention_type == "flash":
                h = self._cross_attention_flash(q, k_extended, v_extended)
            else:
                h = self._cross_attention(q, k_extended, v_extended)
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)

        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h

    def _cross_attention_flash(self, q, k, v):
        # Fallback to manual attention path for cross-attention
        return self._cross_attention(q, k, v)

    def _cross_attention(self, q, k, v):
        bs, q_width, length = q.shape
        ch = q_width // self.num_heads

        q = q.contiguous().view(bs * self.num_heads, ch, length)
        k = k.contiguous().view(bs * self.num_heads, ch, -1)
        v = v.contiguous().view(bs * self.num_heads, ch, -1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, q_width, length)


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention

        assert batch_first
        factory_kwargs = {}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64, 96, 128], "Only support head_dim == 16, 32, 64, 96, 128"

        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads)
        qkv, _ = self.inner_attn(
            qkv.contiguous(),
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    Legacy QKV attention (no flash).
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        from einops import rearrange
        self.rearrange = rearrange

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        qkv = qkv.half()

        qkv = self.rearrange(qkv, "b (three h d) s -> b s three h d", three=3, h=self.n_heads)
        q, k, v = qkv.transpose(1, 3).transpose(3, 4).split(1, dim=2)
        q = q.reshape(bs * self.n_heads, ch, length)
        k = k.reshape(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        a = a.float()
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    Non-flash self-attention (no cross) used by AttentionPool2d.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# -------------------------------
# UNet with global PSF injection to all AttentionBlocks
# -------------------------------
class UNetModel(nn.Module):
    """
    UNet with attention and timestep embedding, conditioned on an intensity map
    and a PSF derived from the k-space mask. The PSF is encoded once and made
    available to EVERY AttentionBlock (down, middle, up) via cross-attention K/V.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        sampling_channels: int = 0,  # kept for compatibility; unused here
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        attention_type="flash",  # can be "flash" or anything else for legacy
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.dtype = th.float16 if use_fp16 else th.float32
        self.model_channels = model_channels
        self.attention_type = attention_type

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # PSF Prompt encoder (produces [B, model_channels, H, W])
        self.prompt_encoder = PromptEncoder(
            in_channels=1,
            embed_dim=model_channels,
            use_conv=True,
            dims=dims,
        )

        # First conv must accept: in_channels + 1 (hr_inte)
        first_in = in_channels + 1
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, first_in, ch, 3, padding=1)
            )
        ])
        input_block_chans = [ch]
        ds = 1

        # Down path
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            prompt_channels=model_channels,  # ensure prompt channels match
                            attention_type=self.attention_type,
                            dims=dims,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                down = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_chans.append(out_ch)
                ch = out_ch
                ds *= 2

        # Middle
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                prompt_channels=model_channels,
                attention_type=self.attention_type,
                dims=dims,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Up path
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            prompt_channels=model_channels,
                            attention_type=self.attention_type,
                            dims=dims,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        (
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=ch)
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final conv
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    # Precision switches
    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    # Forward
    def forward(self, x, timesteps, hr_inte=None, kspace_mask=None, y=None):
        """
        Apply the model to an input batch, conditioned on intensity map and PSF.

        Args:
            x: [B, C, H, W] input image (e.g., noisy image)
            timesteps: [B] diffusion/CM timesteps
            hr_inte: [B, 1, H, W] high-resolution intensity map (required)
            kspace_mask: [B, 1, H, W] k-space sampling mask (optional but recommended)
            y: optional labels (unused)
        """
        if hr_inte is None:
            raise ValueError("hr_inte is required but not provided")

        # Input concat
        x = th.cat([x, hr_inte], dim=1)
        h = x.type(self.dtype)

        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Prompt (PSF) features (same HxW as x; TimestepEmbedSequential will resize as needed)
        prompt_feat = None
        if kspace_mask is not None:
            psf = compute_psf_from_mask(kspace_mask)
            prompt_feat = self.prompt_encoder(psf).to(h.dtype).to(h.device)

        # Down
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, prompt=prompt_feat)
            hs.append(h)

        # Middle
        h = self.middle_block(h, emb, prompt=prompt_feat)

        # Up
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, prompt=prompt_feat)

        h = h.type(x.dtype)
        return self.out(h)
