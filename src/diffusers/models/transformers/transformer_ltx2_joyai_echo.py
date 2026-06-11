# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, logging
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin
from ..cache_utils import CacheMixin
from ..embeddings import PixArtAlphaTextProjection
from ..modeling_utils import ModelMixin
from .transformer_ltx2 import (
    AudioVisualModelOutput,
    LTX2AdaLayerNormSingle,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2VideoTransformerBlock,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class JoyAIEchoTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    A Transformer model for the JoyAI-Echo audiovisual framework, derived from Lightricks LTX-2.3.

    Architecturally this model is identical to
    [`~diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformer3DModel`]; the only delta is in
    [`~JoyAIEchoTransformer3DModel.forward`], which natively plumbs three additional paired-memory attention
    masks (`audio_self_attention_mask`, `a2v_cross_attention_mask`, `v2a_cross_attention_mask`) through to
    each `LTX2VideoTransformerBlock`. The base class hardcodes those three masks to `None`.

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
      activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTX2VideoTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    # Adapted from diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformer3DModel.__init__
    # (body is byte-identical to the base class; only architecturally-shared with no `class`-name references)
    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,  # Video Arguments
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        gated_attn: bool = False,
        cross_attn_mod: bool = False,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        audio_gated_attn: bool = False,
        audio_cross_attn_mod: bool = False,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        use_prompt_embeddings=True,
        perturbed_attn: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        audio_out_channels = audio_out_channels or audio_in_channels
        inner_dim = num_attention_heads * attention_head_dim
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # 1. Patchification input projections
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)

        # 2. Prompt embeddings
        if use_prompt_embeddings:
            # LTX-2.0; LTX-2.3 uses per-modality feature projections in the connector instead
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=audio_inner_dim
            )

        # 3. Timestep Modulation Params and Embedding
        self.prompt_modulation = cross_attn_mod or audio_cross_attn_mod  # used by LTX-2.3

        # 3.1. Global Timestep Modulation Parameters (except for cross-attention) and timestep + size embedding
        # time_embed and audio_time_embed calculate both the timestep embedding and (global) modulation parameters
        video_time_emb_mod_params = 9 if cross_attn_mod else 6
        audio_time_emb_mod_params = 9 if audio_cross_attn_mod else 6
        self.time_embed = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=video_time_emb_mod_params, use_additional_conditions=False
        )
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=audio_time_emb_mod_params, use_additional_conditions=False
        )

        # 3.2. Global Cross Attention Modulation Parameters
        # Used in the audio-to-video and video-to-audio cross attention layers as a global set of modulation params,
        # which are then further modified by per-block modulaton params in each transformer block.
        # There are 2 sets of scale/shift parameters for each modality, 1 each for audio-to-video (a2v) and
        # video-to-audio (v2a) cross attention
        self.av_cross_attn_video_scale_shift = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        self.av_cross_attn_audio_scale_shift = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=4, use_additional_conditions=False
        )
        # Gate param for audio-to-video (a2v) cross attn (where the video is the queries (Q) and the audio is the keys
        # and values (KV))
        self.av_cross_attn_video_a2v_gate = LTX2AdaLayerNormSingle(
            inner_dim, num_mod_params=1, use_additional_conditions=False
        )
        # Gate param for video-to-audio (v2a) cross attn (where the audio is the queries (Q) and the video is the keys
        # and values (KV))
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLayerNormSingle(
            audio_inner_dim, num_mod_params=1, use_additional_conditions=False
        )

        # 3.3. Output Layer Scale/Shift Modulation parameters
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)

        # 3.4. Prompt Scale/Shift Modulation parameters (LTX-2.3)
        if self.prompt_modulation:
            self.prompt_adaln = LTX2AdaLayerNormSingle(inner_dim, num_mod_params=2, use_additional_conditions=False)
            self.audio_prompt_adaln = LTX2AdaLayerNormSingle(
                audio_inner_dim, num_mod_params=2, use_additional_conditions=False
            )

        # 4. Rotary Positional Embeddings (RoPE)
        # Self-Attention
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=inner_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            scale_factors=vae_scale_factors,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=[audio_scale_factor],
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # Audio-to-Video, Video-to-Audio Cross-Attention
        cross_attn_pos_embed_max_pos = max(pos_embed_max_pos, audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="video",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.cross_attn_audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_cross_attention_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )

        # 5. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2VideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    video_gated_attn=gated_attn,
                    video_cross_attn_adaln=cross_attn_mod,
                    audio_gated_attn=audio_gated_attn,
                    audio_cross_attn_adaln=audio_cross_attn_mod,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    perturbed_attn=perturbed_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Output layers
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

        self.gradient_checkpointing = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: torch.LongTensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        video_self_attention_mask: torch.Tensor | None = None,
        audio_self_attention_mask: torch.Tensor | None = None,
        a2v_cross_attention_mask: torch.Tensor | None = None,
        v2a_cross_attention_mask: torch.Tensor | None = None,
        video_cross_attention_sigma: torch.Tensor | None = None,
        audio_cross_attention_sigma: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for the JoyAI-Echo audiovisual video transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input patchified video latents of shape `(batch_size, num_video_tokens, in_channels)`.
            audio_hidden_states (`torch.Tensor`):
                Input patchified audio latents of shape `(batch_size, num_audio_tokens, audio_in_channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Input video text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            audio_encoder_hidden_states (`torch.Tensor`):
                Input audio text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            timestep (`torch.Tensor`):
                Input timestep of shape `(batch_size, num_video_tokens)`. These should already be scaled by
                `self.config.timestep_scale_multiplier`.
            audio_timestep (`torch.Tensor`, *optional*):
                Input timestep of shape `(batch_size,)` or `(batch_size, num_audio_tokens)` for audio modulation
                params. This is only used by certain pipelines such as the I2V pipeline.
            sigma (`torch.Tensor`, *optional*):
                Input scaled timestep of shape (batch_size,). Used for video prompt cross attention modulation in
                models such as LTX-2.3.
            audio_sigma (`torch.Tensor`, *optional*):
                Input scaled timestep of shape (batch_size,). Used for audio prompt cross attention modulation in
                models such as LTX-2.3. If `sigma` is supplied but `audio_sigma` is not, `audio_sigma` will be set to
                the provided `sigma` value.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
            audio_encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)` for audio modeling.
            num_frames (`int`, *optional*):
                The number of latent video frames. Used if calculating the video coordinates for RoPE.
            height (`int`, *optional*):
                The latent video height. Used if calculating the video coordinates for RoPE.
            width (`int`, *optional*):
                The latent video width. Used if calculating the video coordinates for RoPE.
            fps (`float`, *optional*, defaults to `24.0`):
                Video frames per second. Used if calculating the video coordinates for RoPE.
            audio_num_frames (`int`, *optional*):
                The number of audio frames. Used if calculating the audio coordinates for RoPE.
            video_coords (`torch.Tensor`, *optional*):
                Precomputed video coordinates for RoPE.
            audio_coords (`torch.Tensor`, *optional*):
                Precomputed audio coordinates for RoPE.
            isolate_modalities (`bool`, *optional*, defaults to `False`):
                If `True`, disables the audio-to-video and video-to-audio cross attention layers, effectively isolating
                the audio and video modalities within each transformer block.
            spatio_temporal_guidance_blocks (`list[int]`, *optional*):
                Indices of the transformer blocks to apply spatio-temporal guidance (STG) at, simulating the
                self-attention operations by simply using the values rather than the full scaled dot-product attention
                (SDPA) operation. If `None` or empty, STG will not be applied to any block.
            perturbation_mask (`torch.Tensor`, *optional*):
                Perturbation mask for STG of shape `(batch_size,)` or `(batch_size, 1, 1)`. Should be 0 at batch
                elements where STG should be applied and 1 elsewhere. If STG is being used but `peturbation_mask` is
                not supplied, will default to applying STG (perturbing) all batch elements.
            use_cross_timestep (`bool` *optional*, defaults to `False`):
                Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
                calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
                `False` is the legacy LTX-2.0 behavior.
            attention_kwargs (`dict[str, Any]`, *optional*):
                Optional dict of keyword args to be passed to the attention processor.
            video_self_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative self-attention mask of shape `(batch_size, num_video_tokens, num_video_tokens)`
                applied to the video self-attention in each transformer block. Values in `[0, 1]` where `1` means full
                attention and `0` means masked. Used e.g. by the IC-LoRA pipeline to control attention strength between
                noisy tokens and appended reference tokens. Audio self-attention is not affected.
            audio_self_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative self-attention mask of shape `(batch_size, num_audio_tokens, num_audio_tokens)`
                applied to the audio self-attention in each transformer block. Values in `[0, 1]` where `1` means full
                attention and `0` means masked. This is the JoyAI-Echo paired-memory audio mask not exposed by the base
                LTX-2 transformer.
            a2v_cross_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative attention mask of shape `(batch_size, num_video_tokens, num_audio_tokens)`
                applied to the audio-to-video (a2v) cross attention (Q: video; K, V: audio) in each transformer block.
                Values in `[0, 1]` where `1` means full attention and `0` means masked. This is one of the JoyAI-Echo
                paired-memory masks not exposed by the base LTX-2 transformer.
            v2a_cross_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative attention mask of shape `(batch_size, num_audio_tokens, num_video_tokens)`
                applied to the video-to-audio (v2a) cross attention (Q: audio; K, V: video) in each transformer block.
                Values in `[0, 1]` where `1` means full attention and `0` means masked. This is one of the JoyAI-Echo
                paired-memory masks not exposed by the base LTX-2 transformer.
            video_cross_attention_sigma (`torch.Tensor`, *optional*):
                Scalar or per-batch sigma of shape `(batch_size,)` or `(batch_size, 1)` used to derive the AdaLN
                modulation parameters for the VIDEO cross-attention (a2v: Q: video; K, V: audio). When supplied,
                this REPLACES the per-token `timestep` / `audio_sigma` that would otherwise feed the cross-attn
                modulation, so memory-prefix tokens (which carry σ=0 in the per-token `timestep`) do not pollute
                the cross-attn modulation. This is required for JoyAI-Echo memory injection: the source wrapper
                feeds the SCALAR `Modality.sigma` (target σ) to the cross-attn AdaLN regardless of the per-token
                σ=0 prefix used for self-attn AdaLN. When `None`, falls back to the legacy behavior (uses the
                per-token `timestep` / `audio_sigma` selected by `use_cross_timestep`), preserving bit-exact
                base LTX-2 behavior.
            audio_cross_attention_sigma (`torch.Tensor`, *optional*):
                Same as `video_cross_attention_sigma` but for the AUDIO cross-attention (v2a: Q: audio; K, V:
                video) modulation. Replaces the per-token `audio_timestep` / `sigma` that would otherwise feed
                the cross-attn modulation. When `None`, falls back to the legacy behavior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dict-like structured output of type `AudioVisualModelOutput` or a tuple.

        Returns:
            `AudioVisualModelOutput` or `tuple`:
                If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
                `tuple` is returned where the first element is the denoised video latent patch sequence and the second
                element is the denoised audio latent patch sequence.
        """
        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        # Convert video_self_attention_mask from multiplicative mask ([0, 1]) to additive bias form (0 / -10000)
        # matching the encoder_attention_mask convention above. Shape is preserved: (B, T_v, T_v).
        if video_self_attention_mask is not None:
            video_self_attention_mask = (1 - video_self_attention_mask.to(hidden_states.dtype)) * -10000.0

        # JoyAI-Echo paired-memory masks: convert from multiplicative ([0, 1]) to additive bias form, mirroring the
        # video_self_attention_mask handling above. These are passed through to each LTX2VideoTransformerBlock.
        if audio_self_attention_mask is not None:
            audio_self_attention_mask = (1 - audio_self_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
        if a2v_cross_attention_mask is not None:
            a2v_cross_attention_mask = (1 - a2v_cross_attention_mask.to(hidden_states.dtype)) * -10000.0
        if v2a_cross_attention_mask is not None:
            v2a_cross_attention_mask = (1 - v2a_cross_attention_mask.to(audio_hidden_states.dtype)) * -10000.0

        batch_size = hidden_states.size(0)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
        )

        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the output layer
        # modulation with scale_shift_table (and similarly for audio)
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        if self.prompt_modulation:
            # LTX-2.3
            temb_prompt, _ = self.prompt_adaln(
                sigma.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
        else:
            temb_prompt = temb_prompt_audio = None

        # 3.2. Prepare global modality cross attention modulation parameters
        # JoyAI-Echo memory injection passes a scalar `video_cross_attention_sigma` to override the per-token
        # `timestep` / `audio_sigma` that would otherwise feed the AdaLN here. With per-token timesteps that
        # carry σ=0 over the memory prefix (correct for self-attn modulation), feeding them through cross-attn
        # AdaLN would silently mis-modulate cross-attn over the memory tokens. The source wrapper uses the
        # SCALAR `Modality.sigma` (target σ) for cross-attn AdaLN regardless of the σ=0 prefix
        # (`ltx_core/model/transformer/transformer_args.py:279-303` in the JoyAI-Echo source).
        # When the override is `None`, fall back to today's behavior (bit-exact base LTX-2). C+D.2b P1 fix.
        if video_cross_attention_sigma is not None:
            video_ca_timestep = video_cross_attention_sigma.flatten()
        else:
            video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])

        if audio_cross_attention_sigma is not None:
            audio_ca_timestep = audio_cross_attention_sigma.flatten()
        else:
            audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])

        # 4. Prepare prompt embeddings (LTX-2.0)
        if self.config.use_prompt_embeddings:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # 5. Run transformer blocks
        spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
        if len(spatio_temporal_guidance_blocks) > 0 and perturbation_mask is None:
            # If STG is being used and perturbation_mask is not supplied, default to perturbing all batch elements.
            perturbation_mask = torch.zeros((batch_size,))
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]  # unsqueeze to 3D to broadcast with hidden_states
        all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
        stg_blocks = set(spatio_temporal_guidance_blocks)

        for block_idx, block in enumerate(self.transformer_blocks):
            block_perturbation_mask = perturbation_mask if block_idx in stg_blocks else None
            block_all_perturbed = all_perturbed if block_idx in stg_blocks else False

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    temb_prompt,
                    temb_prompt_audio,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    video_self_attention_mask,  # self_attention_mask (video-only)
                    audio_self_attention_mask,  # JoyAI-Echo paired-memory mask
                    a2v_cross_attention_mask,  # JoyAI-Echo paired-memory mask
                    v2a_cross_attention_mask,  # JoyAI-Echo paired-memory mask
                    not isolate_modalities,  # use_a2v_cross_attention
                    not isolate_modalities,  # use_v2a_cross_attention
                    block_perturbation_mask,
                    block_all_perturbed,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    self_attention_mask=video_self_attention_mask,
                    audio_self_attention_mask=audio_self_attention_mask,
                    a2v_cross_attention_mask=a2v_cross_attention_mask,
                    v2a_cross_attention_mask=v2a_cross_attention_mask,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=block_perturbation_mask,
                    all_perturbed=block_all_perturbed,
                )

        # 6. Output layers (including unpatchification)
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
