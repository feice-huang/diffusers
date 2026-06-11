# Copyright 2025 Lightricks, JoyAI, and The HuggingFace Team. All rights reserved.
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

import copy  # noqa: F401  # retained for parity with LTX2Pipeline; used by B.3 __call__
import inspect
from typing import Any, Callable, Literal, Optional

import numpy as np  # noqa: F401  # retained for parity with LTX2Pipeline; used by B.3 __call__
import torch
import torchvision.transforms.functional as TVF
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor, GemmaTokenizer, GemmaTokenizerFast

from ...callbacks import MultiPipelineCallbacks, PipelineCallback  # noqa: F401  # parity, used in B.3
from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
from ...models.transformers.transformer_ltx2_joyai_echo import JoyAIEchoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..ltx2.connectors import LTX2TextConnectors
from ..ltx2.pipeline_output import LTX2PipelineOutput
from ..ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from ..pipeline_utils import DiffusionPipeline
from .memory_bank import JoyAIEchoMemoryBank


# JoyAI-Echo reuses the LTX-2 pipeline output dataclass (multi-shot outputs are concatenated
# along the time axis and exposed via the same `frames`/`audio` fields).
JoyAIEchoPipelineOutput = LTX2PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # noqa: F401  # parity with LTX2Pipeline

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:

    ## Default schedule (40-step LTX-2.3 baseline)

    ```py
    >>> import torch
    >>> from diffusers import JoyAIEchoPipeline

    >>> pipe = JoyAIEchoPipeline.from_pretrained("jdopensource/JoyAI-Echo", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> output = pipe(
    ...     prompt="A close-up of a violinist playing on a candlelit stage.",
    ...     height=512,
    ...     width=768,
    ...     num_frames=121,
    ...     num_inference_steps=40,
    ...     guidance_scale=4.0,
    ... )
    ```

    ## DMD 8-step inference (~7.5x faster, CFG-free)

    JoyAI-Echo also ships a Distribution-Matching-Distillation (DMD) checkpoint that converges in 8
    denoising steps. Pass the 9-sigma ladder exposed on the class via `sigmas=` (drop the trailing
    `0.0` so the scheduler runs the source's 8 transitions instead of an extra no-op forward), set
    `guidance_scale=1.0` to disable classifier-free guidance (the pipeline auto-gates CFG on
    `guidance_scale > 1.0`, so this skips the conditional-batch concat), and leave
    `num_inference_steps` to be derived from the sigma list:

    ```py
    >>> import torch
    >>> from diffusers import JoyAIEchoPipeline

    >>> pipe = JoyAIEchoPipeline.from_pretrained("jdopensource/JoyAI-Echo", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> output = pipe(
    ...     prompt="A close-up of a violinist playing on a candlelit stage.",
    ...     height=512,
    ...     width=768,
    ...     num_frames=121,
    ...     num_inference_steps=None,
    ...     sigmas=list(JoyAIEchoPipeline.DMD_SIGMAS)[:-1],  # 8 transitions, matches source
    ...     guidance_scale=1.0,  # DMD is CFG-free
    ...     audio_guidance_scale=1.0,
    ... )
    ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class JoyAIEchoPipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for JoyAI-Echo long multi-shot audio-video generation, built on top of Lightricks LTX-2.3.

    This is the SCAFFOLD class (slice B.1). It mirrors the structure of `LTX2Pipeline` so that all helper methods
    (text encoding, latent packing, normalization, etc.) remain byte-identical via `# Copied from` markers, while
    `__init__` accepts the JoyAI-Echo transformer (`JoyAIEchoTransformer3DModel`) instead of the base
    `LTX2VideoTransformer3DModel`. The `__call__` method is intentionally a `NotImplementedError` stub; subsequent
    slices fill it in:

      * B.3 — single-shot `__call__` implementation
      * C   — video memory via `video_self_attention_mask`
      * D   — audio memory via `audio_self_attention_mask` and paired a2v/v2a masks
      * E   — DMD 9-sigma schedule integration

    Args:
        transformer ([`JoyAIEchoTransformer3DModel`]):
            JoyAI-Echo Transformer architecture to denoise the encoded multi-shot video/audio latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTX2Video`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Gemma3ForConditionalGeneration`]):
            Gemma3 text encoder used by LTX-2.3.
        tokenizer (`GemmaTokenizer` | `GemmaTokenizerFast`):
            Gemma tokenizer matching `text_encoder`.
        connectors ([`LTX2TextConnectors`]):
            Text connector stack used to adapt text encoder hidden states for the video and audio branches.
            JoyAI-Echo configures this with `per_modality_projections=True, video_hidden_dim=4096,
            audio_hidden_dim=2048, proj_bias=True, video_connector_num_layers=8, audio_connector_num_layers=8,
            video_gated_attn=True, audio_gated_attn=True` at instantiation time.
    """

    model_cpu_offload_seq = "text_encoder->connectors->transformer->vae->audio_vae->vocoder"
    _optional_components = ["processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    # DMD (Distribution-Matching Distillation) 9-sigma schedule. JoyAI-Echo ships a distilled checkpoint
    # that converges in 8 denoising steps (9 sigmas -> 8 steps in the Flow-Matching Euler scheduler).
    # Sigma values verbatim from the source:
    #   ltx-pipelines/src/ltx_pipelines/utils/constants.py:15 (DISTILLED_SIGMA_VALUES)
    #   configs/inference.yaml:48-57 (denoising.sigmas + denoising.steps)
    # Pass via the `sigmas=` kwarg of `__call__` together with `guidance_scale=1.0` (DMD is CFG-free,
    # so the pipeline's auto-gated `do_classifier_free_guidance` property skips the conditional-concat
    # / batch-doubling and yields the expected ~7.5x throughput vs. the default 60-step schedule).
    DMD_SIGMAS: tuple[float, ...] = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
    DMD_TIMESTEPS: tuple[int, ...] = (1000, 994, 988, 981, 975, 909, 725, 422, 0)

    # Adapted from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.__init__
    # The only change from the base is the type annotation on `transformer`, which is now the JoyAI-Echo
    # variant rather than the base LTX2VideoTransformer3DModel. The body is byte-identical to the base.
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast,
        connectors: LTX2TextConnectors,
        transformer: JoyAIEchoTransformer3DModel,
        vocoder: LTX2Vocoder | LTX2VocoderWithBWE,
        processor: Gemma3Processor | None = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
            scheduler=scheduler,
            processor=processor,
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        # TODO: check whether the MEL compression ratio logic here is corrct
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer") is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if getattr(self, "tokenizer", None) is not None else 1024
        )

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._get_gemma_prompt_embeds
    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            device: (`str` or `torch.device`):
                torch device to place the resulting embeddings on
            dtype: (`torch.dtype`):
                torch dtype to cast the prompt embeds to
            max_sequence_length (`int`, defaults to 1024): Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if getattr(self, "tokenizer", None) is not None:
            # Gemma expects left padding for chat-style prompts
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        prompt_embeds = text_encoder_hidden_states.flatten(2, 3).to(dtype=dtype)  # Pack to 3D

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copy of diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.enhance_prompt (the `@torch.no_grad()` decorator
    # placement makes the automated `# Copied from` machinery unreliable for this method — its `enhance_prompt`
    # body is byte-identical to the base; keep them in sync manually when the base changes).
    @torch.no_grad()
    def enhance_prompt(
        self,
        prompt: str,
        system_prompt: str,
        max_new_tokens: int = 512,
        seed: int = 10,
        generator: torch.Generator | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Enhances the supplied `prompt` by generating a new prompt using the current text encoder (default is a
        `transformers.Gemma3ForConditionalGeneration` model) from it and a system prompt.
        """
        device = device or self._execution_device
        if generation_kwargs is None:
            # Set to default generation kwargs
            generation_kwargs = {"do_sample": True, "temperature": 0.7}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        template = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(text=template, images=None, return_tensors="pt").to(device)
        self.text_encoder.to(device)

        # `transformers.GenerationMixin.generate` does not support using a `torch.Generator` to control randomness,
        # so manually apply a seed for reproducible generation.
        if generator is not None:
            # Overwrite seed to generator's initial seed
            seed = generator.initial_seed()
        torch.manual_seed(seed)
        generated_sequences = self.text_encoder.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )  # tensor of shape [batch_size, seq_len]

        generated_ids = [seq[len(model_inputs.input_ids[i]) :] for i, seq in enumerate(generated_sequences)]
        enhanced_prompt = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return enhanced_prompt

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
        audio_stg_scale=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if ((stg_scale > 0.0) or (audio_stg_scale > 0.0)) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is specified but no STG blocks are supplied. Please supply a list of"
                "block indices at which to apply STG in `spatio_temporal_guidance_blocks`"
            )

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_latents
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._normalize_latents
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._normalize_audio_latents
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_audio_latents
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._create_noised_state
    def _create_noised_state(
        latents: torch.Tensor, noise_scale: float | torch.Tensor, generator: torch.Generator | None = None
    ):
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised_latents = noise_scale * noise + (1 - noise_scale) * latents
        return noised_latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._pack_audio_latents
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
    ) -> torch.Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if patch_size is not None and patch_size_t is not None:
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents = latents.transpose(1, 2).flatten(2, 3)  # [B, C, L, M] --> [B, L, C * M]
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_audio_latents
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    @staticmethod
    def _memory_slot_ranges(total_seq_len: int, num_slots: int) -> list[tuple[int, int]]:
        """Equal-division of ``[0, total_seq_len)`` into ``num_slots`` half-open ranges.

        Mirrors ``LTX2DiffusionWrapper._memory_slot_ranges`` at
        ``ltx-distillation/src/ltx_distillation/models/ltx_wrapper.py:342-354``. Used for the dimension
        that has no per-slot length metadata (the video dimension on both paired cross-attention masks).
        """
        if total_seq_len <= 0 or num_slots <= 0:
            return []
        ranges: list[tuple[int, int]] = []
        start = 0
        for slot_idx in range(num_slots):
            end = round((slot_idx + 1) * total_seq_len / num_slots)
            if end > start:
                ranges.append((start, end))
            start = end
        return ranges

    @staticmethod
    def _memory_slot_ranges_from_lengths(
        lengths: tuple[int, ...] | None,
        total_seq_len: int,
        num_slots: int,
    ) -> list[tuple[int, int]]:
        """Per-slot ranges from explicit lengths, with equal-division fallback.

        Mirrors ``LTX2DiffusionWrapper._memory_slot_ranges_from_lengths`` at
        ``ltx-distillation/src/ltx_distillation/models/ltx_wrapper.py:356-376``. If ``lengths`` is
        missing, has the wrong slot count, or does not sum exactly to ``total_seq_len``, falls back to
        ``_memory_slot_ranges`` (equal-division).
        """
        if not lengths or len(lengths) != num_slots:
            return JoyAIEchoPipeline._memory_slot_ranges(total_seq_len, num_slots)
        ranges: list[tuple[int, int]] = []
        start = 0
        for raw_length in lengths:
            length = max(0, int(raw_length))
            end = min(start + length, total_seq_len)
            if end > start:
                ranges.append((start, end))
            start = end
        if start != total_seq_len:
            return JoyAIEchoPipeline._memory_slot_ranges(total_seq_len, num_slots)
        return ranges

    @staticmethod
    def _build_audio_self_attention_mask(
        batch_size: int,
        memory_seq_len: int,
        target_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Audio self-attention block mask for ``[memory | target]`` audio tokens.

        Mirrors ``LTX2DiffusionWrapper._build_memory_self_attention_block_mask`` at
        ``ltx-distillation/src/ltx_distillation/models/ltx_wrapper.py:430-452``. Returns a float mask
        of shape ``(batch_size, T_a_total, T_a_total)`` where ``T_a_total = memory_seq_len +
        target_seq_len``. Convention: ``1.0 = keep``, ``0.0 = mask`` (the JoyAI-Echo transformer
        converts this to additive bias via ``(1 - mask) * -10000`` at
        ``transformer_ltx2_joyai_echo.py:463-464``).

        Block structure (spec D.0 §3): ``noisy ↔ noisy = 1`` (full block over
        ``[T_a_mem:, T_a_mem:]``), ``mem ↔ mem = 1`` (full block over ``[:T_a_mem, :T_a_mem]``,
        ALL-to-ALL within memory — no per-slot grouping), and both cross-blocks (``mem → noisy`` and
        ``noisy → mem``) are 0.
        """
        if memory_seq_len <= 0:
            return None
        total_seq_len = memory_seq_len + target_seq_len
        mask = torch.ones(batch_size, total_seq_len, total_seq_len, device=device, dtype=dtype)
        # Source builds with bool, then sets cross-blocks False, then re-enables mem↔mem. Same result.
        mask[:, :, :memory_seq_len] = 0.0
        mask[:, :memory_seq_len, :] = 0.0
        mask[:, :memory_seq_len, :memory_seq_len] = 1.0
        return mask

    @staticmethod
    def _build_paired_memory_cross_mask(
        batch_size: int,
        query_memory_seq_len: int,
        query_target_seq_len: int,
        kv_memory_seq_len: int,
        kv_target_seq_len: int,
        num_memory_slots: int,
        device: torch.device,
        dtype: torch.dtype,
        query_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
        kv_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
    ) -> torch.Tensor:
        """Per-slot paired cross-attention mask between memory + target tokens of two modalities.

        Mirrors ``LTX2DiffusionWrapper._build_paired_memory_cross_mask`` at
        ``ltx-distillation/src/ltx_distillation/models/ltx_wrapper.py:378-428``. Returns a float mask
        of shape ``(batch_size, T_q_total, T_kv_total)`` (``T_q_total = q_mem + q_tgt`` etc.).
        Convention: ``1.0 = keep``, ``0.0 = mask``.

        Block structure (spec D.0 §4):
        - ``target ↔ target = 1`` (full block over ``[q_mem:, kv_mem:]``).
        - ``mem_slot_i ↔ mem_slot_i = 1`` (paired per slot — cross-slot mem talk is blocked).
        - everything else = 0 (target queries cannot pull from memory KV, memory queries cannot pull
          from target KV).

        ``*_segment_lengths`` follow the source's per-batch nesting (outer tuple over batches, inner
        tuple over slots); see ``memory_bank.get_memory_audio_segment_lengths``.
        """
        query_total_seq_len = query_memory_seq_len + query_target_seq_len
        kv_total_seq_len = kv_memory_seq_len + kv_target_seq_len
        mask = torch.zeros(batch_size, query_total_seq_len, kv_total_seq_len, device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            query_lengths = (
                query_segment_lengths[batch_idx]
                if query_segment_lengths is not None and batch_idx < len(query_segment_lengths)
                else None
            )
            kv_lengths = (
                kv_segment_lengths[batch_idx]
                if kv_segment_lengths is not None and batch_idx < len(kv_segment_lengths)
                else None
            )
            query_ranges = JoyAIEchoPipeline._memory_slot_ranges_from_lengths(
                query_lengths, query_memory_seq_len, num_memory_slots
            )
            kv_ranges = JoyAIEchoPipeline._memory_slot_ranges_from_lengths(
                kv_lengths, kv_memory_seq_len, num_memory_slots
            )
            for (q_start, q_end), (k_start, k_end) in zip(query_ranges, kv_ranges, strict=False):
                mask[batch_idx, q_start:q_end, k_start:k_end] = 1.0
        if query_target_seq_len > 0 and kv_target_seq_len > 0:
            mask[:, query_memory_seq_len:, kv_memory_seq_len:] = 1.0
        return mask

    @staticmethod
    def _compute_mel_for_audio_selection(
        waveform: torch.Tensor,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        mel_bins: int,
    ) -> torch.Tensor:
        """Compute a log-mel spectrogram for audio-window selection.

        Mirrors ``AudioProcessor.waveform_to_mel`` at
        ``ltx-core/src/ltx_core/model/audio_vae/ops.py:47-58``: returns a 4-D
        ``[B, C, T_mel, mel_bins]`` layout (T at dim=2) so the downstream selector
        can score with per-channel ``exp().sum(dim=(1,2,3))`` matching the source.
        ``waveform`` may be 1-D ``[T]``, 2-D ``[C, T]`` / ``[B, T]``, or 3-D
        ``[B, C, T]``.
        """
        import torchaudio

        wav = waveform.detach()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)
        elif wav.dim() != 3:
            raise ValueError(f"`waveform` must have 1, 2, or 3 dims, got shape={tuple(wav.shape)}.")

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=int(n_fft),
            win_length=int(n_fft),
            hop_length=int(hop_length),
            f_min=0.0,
            f_max=float(sample_rate) / 2.0,
            n_mels=int(mel_bins),
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device=wav.device, dtype=torch.float32)

        mel = mel_transform(wav.float())  # [B, C, mel_bins, T_mel]
        mel = torch.log(torch.clamp(mel, min=1e-5))
        # Match source: permute to [B, C, T_mel, mel_bins] (T at dim=2).
        return mel.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _select_audio_window_with_bounds(
        mel: torch.Tensor,
        window_size_latent: int,
        downsample_factor: int,
        selection_mode: Literal["max_response", "random", "center"] = "max_response",
        is_causal: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[int, int, int, int]:
        """Pick an audio window from a 4-D mel spectrogram.

        Mirrors ``select_audio_window_with_bounds`` at
        ``ltx-distillation/src/ltx_distillation/audio_memory.py:124-135`` together
        with ``select_max_response_audio_window_with_bounds`` at
        ``audio_memory.py:30-71`` and the latent<->pixel translation at lines 13-27.
        Expects ``mel`` shape ``[B, C, T_mel, mel_bins]`` (matching the source's
        post-``permute(0, 1, 3, 2)`` layout). Returns ``(latent_start, latent_end,
        mel_start, mel_end)`` half-open indices for the first batch element.
        """
        if mel.dim() != 4:
            raise ValueError(f"`mel` must have shape [B, C, T_mel, mel_bins], got {tuple(mel.shape)}.")
        if window_size_latent <= 0:
            raise ValueError(f"`window_size_latent` must be positive, got {window_size_latent}.")
        if downsample_factor <= 0:
            raise ValueError(f"`downsample_factor` must be positive, got {downsample_factor}.")

        mode = str(selection_mode).lower()
        if mode not in {"max_response", "random", "center"}:
            raise ValueError(f"Unsupported selection_mode: {selection_mode!r}.")

        pixel_window_size = int(window_size_latent) * int(downsample_factor)
        if is_causal:
            pixel_window_size = max(pixel_window_size - (int(downsample_factor) - 1), 1)

        num_time_steps = mel.shape[2]
        # `is_causal` only shrinks `pixel_window_size` above (per source
        # `latent_window_size_to_pixel_window_size`); it does NOT clip the
        # searchable range. The full spectrogram remains scannable.
        max_start_idx = max(0, num_time_steps - pixel_window_size)

        if mode == "max_response":
            scan_stride = max(1, pixel_window_size // 4)
            candidates = list(range(0, max_start_idx + 1, scan_stride))
            if candidates and candidates[-1] != max_start_idx:
                candidates.append(max_start_idx)
            if not candidates:
                candidates = [0]
            offsets = torch.arange(pixel_window_size, device=mel.device)
            scores = []
            for start in candidates:
                idx = (start + offsets).clamp(0, num_time_steps - 1).long()
                window = mel.index_select(dim=2, index=idx)
                # Per-channel exp then sum over (C, T-in-window, F) -- matches
                # `audio_memory.py:57` `window.float().exp().sum(dim=(1, 2, 3))`.
                scores.append(window.float().exp().sum(dim=(1, 2, 3)))
            stacked = torch.stack(scores, dim=1)  # [B, num_candidates]
            best = int(stacked[0].argmax().item())
            mel_start = int(candidates[best])
        elif mode == "random":
            if max_start_idx <= 0:
                mel_start = 0
            else:
                rand = torch.randint(
                    low=0,
                    high=max_start_idx + 1,
                    size=(1,),
                    generator=generator,
                    device=generator.device if generator is not None else "cpu",
                )
                mel_start = int(rand.item())
        else:  # center
            mel_start = max_start_idx // 2

        mel_end = min(mel_start + pixel_window_size, num_time_steps)
        # Translate mel-frame coords back to latent coords via the same
        # downsample factor used to derive `pixel_window_size`.
        latent_start = mel_start // int(downsample_factor)
        latent_end = latent_start + int(window_size_latent)
        return int(latent_start), int(latent_end), int(mel_start), int(mel_end)

    @staticmethod
    def _mel_window_bounds_to_seconds(
        mel_start: int,
        mel_end: int,
        hop_length: int,
        sample_rate: int,
    ) -> tuple[float, float]:
        """Convert mel-frame indices to seconds.

        Mirrors ``mel_window_bounds_to_seconds`` at
        ``ltx-distillation/src/ltx_distillation/audio_memory.py:138-156``. Uses the
        pure formula ``t = mel_idx * hop_length / sample_rate`` for both endpoints
        (half-open ``[mel_start, mel_end)`` translates to ``[t_start, t_end)``).
        """
        if mel_start < 0:
            raise ValueError(f"`mel_start` must be non-negative, got {mel_start}.")
        if mel_end < mel_start:
            raise ValueError(f"`mel_end` must be >= mel_start, got start={mel_start}, end={mel_end}.")
        if hop_length <= 0:
            raise ValueError(f"`hop_length` must be positive, got {hop_length}.")
        if sample_rate <= 0:
            raise ValueError(f"`sample_rate` must be positive, got {sample_rate}.")
        t_start = float(mel_start * hop_length) / float(sample_rate)
        t_end = float(mel_end * hop_length) / float(sample_rate)
        return t_start, t_end

    @staticmethod
    def _select_video_frame_indices_from_time_range(
        t_start: float,
        t_end: float,
        video_fps: float,
        total_video_frames: int,
        clip_num_frames: int,
    ) -> list[int]:
        """Return ``clip_num_frames`` integer frame indices centered on the given time range.

        Mirrors ``select_video_frame_indices_from_time_range`` at
        ``ltx-distillation/src/ltx_distillation/audio_memory.py:159-212`` but in
        a single ``"center"``-style call: build the candidate range
        ``[ceil(t_start*fps), ceil(t_end*fps)-1]``, then take a centered slice of
        ``clip_num_frames`` items. All indices are clamped to
        ``[0, total_video_frames)``.
        """
        import math

        if total_video_frames <= 0:
            raise ValueError(f"`total_video_frames` must be positive, got {total_video_frames}.")
        if video_fps <= 0:
            raise ValueError(f"`video_fps` must be positive, got {video_fps}.")
        if clip_num_frames <= 0:
            raise ValueError(f"`clip_num_frames` must be positive, got {clip_num_frames}.")
        if t_end < t_start:
            raise ValueError(f"`t_end` must be >= t_start, got ({t_start}, {t_end}).")

        start_frame = int(math.ceil(float(t_start) * float(video_fps)))
        end_frame = int(math.ceil(float(t_end) * float(video_fps))) - 1
        start_frame = max(0, min(start_frame, total_video_frames - 1))
        end_frame = max(0, min(end_frame, total_video_frames - 1))

        if end_frame < start_frame:
            center_time = max(0.0, 0.5 * (float(t_start) + float(t_end)))
            center_frame = int(round(center_time * float(video_fps)))
            candidates = [max(0, min(center_frame, total_video_frames - 1))]
        else:
            candidates = list(range(start_frame, end_frame + 1))

        if len(candidates) <= clip_num_frames:
            selected = candidates[:]
        else:
            offset = max(0, (len(candidates) - clip_num_frames) // 2)
            selected = candidates[offset : offset + clip_num_frames]

        if len(selected) < clip_num_frames:
            pad = selected[-1] if selected else 0
            selected = selected + [pad] * (clip_num_frames - len(selected))
        return [int(i) for i in selected]

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 5:
                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                # latents are of shape [B, C, F, H, W], need to be packed
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            if latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is [batch_size, num_seq, num_features]."
                )
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        return latents

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.prepare_audio_latents
    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,  # 1 is just a dummy value
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 4:
                # latents are of shape [B, C, L, M], need to be packed
                latents = self._pack_audio_latents(latents)
            if latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is [batch_size, num_seq, num_features]."
                )
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        shape = (batch_size, num_channels_latents, audio_latent_length, latent_mel_bins)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_audio_latents(latents)
        return latents

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.convert_velocity_to_x0
    def convert_velocity_to_x0(
        self, sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: Any | None = None
    ) -> torch.Tensor:
        if scheduler is None:
            scheduler = self.scheduler

        sample_x0 = sample - denoised_output * scheduler.sigmas[step_idx]
        return sample_x0

    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.convert_x0_to_velocity
    def convert_x0_to_velocity(
        self, sample: torch.Tensor, denoised_output: torch.Tensor, step_idx: int, scheduler: Any | None = None
    ) -> torch.Tensor:
        if scheduler is None:
            scheduler = self.scheduler

        sample_v = (sample - denoised_output) / scheduler.sigmas[step_idx]
        return sample_v

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def stg_scale(self):
        return self._stg_scale

    @property
    def modality_scale(self):
        return self._modality_scale

    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def audio_guidance_rescale(self):
        return self._audio_guidance_rescale

    @property
    def audio_stg_scale(self):
        return self._audio_stg_scale

    @property
    def audio_modality_scale(self):
        return self._audio_modality_scale

    @property
    def do_classifier_free_guidance(self):
        return (self._guidance_scale > 1.0) or (self._audio_guidance_scale > 1.0)

    @property
    def do_spatio_temporal_guidance(self):
        return (self._stg_scale > 0.0) or (self._audio_stg_scale > 0.0)

    @property
    def do_modality_isolation_guidance(self):
        return (self._modality_scale > 1.0) or (self._audio_modality_scale > 1.0)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    # Adapted from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.__call__
    # The body matches the base pipeline byte-for-byte UNTIL a non-empty `memory_bank` is supplied.
    # When memory is present, this method prefix-prepends VAE-encoded video memory and (if available)
    # paired audio memory to each transformer call, with per-token σ=0 over the prefix and the prefix
    # stripped from the transformer output before the scheduler step. Mask construction (paired a2v /
    # v2a / audio self-attention masks) is C+D.1c's job and remains unset here. The B.4 single-shot
    # parity contract is preserved when `memory_bank is None` or `len(memory_bank) == 0`.
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        sigmas: list[float] | None = None,
        timesteps: list[int] = None,
        guidance_scale: float = 4.0,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        audio_guidance_scale: float | None = None,
        audio_stg_scale: float | None = None,
        audio_modality_scale: float | None = None,
        audio_guidance_rescale: float | None = None,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = False,
        system_prompt: str | None = None,
        prompt_max_new_tokens: int = 512,
        prompt_enhancement_kwargs: dict[str, Any] | None = None,
        prompt_enhancement_seed: int = 10,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
        memory_bank: Optional[JoyAIEchoMemoryBank] = None,
        memory_position_mode: Literal["legacy", "prefix_continuous"] = "legacy",
        memory_downscale_factor: int = 1,
        paired_audio_memory: bool = True,
        audio_memory_window_size: int = 96,
        audio_memory_window_selection_mode: Literal["max_response", "random", "center"] = "max_response",
        audio_memory_mel_bins: int = 128,
        audio_memory_mel_hop_length: int = 160,
        audio_memory_n_fft: int = 1024,
        audio_memory_downsample_factor: int = 4,
        audio_memory_is_causal: bool = True,
        audio_memory_sample_rate: int = 16000,
        video_memory_clip_num_frames: int = 9,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            height (`int`, *optional*, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to `768`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, *optional*, defaults to `121`):
                The number of video frames to generate
            frame_rate (`float`, *optional*, defaults to `24.0`):
                The frames per second (FPS) of the generated video.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            timesteps (`list[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to `4.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. Used for the video modality (there is
                a separate value `audio_guidance_scale` for the audio modality).
            stg_scale (`float`, *optional*, defaults to `0.0`):
                Video guidance scale for Spatio-Temporal Guidance (STG), proposed in [Spatiotemporal Skip Guidance for
                Enhanced Video Diffusion Sampling](https://arxiv.org/abs/2411.18664). STG uses a CFG-like estimate
                where we move the sample away from a weak sample from a perturbed version of the denoising model.
                Enabling STG will result in an additional denoising model forward pass; the default value of `0.0`
                means that STG is disabled.
            modality_scale (`float`, *optional*, defaults to `1.0`):
                Video guidance scale for LTX-2.X modality isolation guidance, where we move the sample away from a
                weaker sample generated by the denoising model withy cross-modality (audio-to-video and video-to-audio)
                cross attention disabled using a CFG-like estimate. Enabling modality guidance will result in an
                additional denoising model forward pass; the default value of `1.0` means that modality guidance is
                disabled.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR. Used for the video modality.
            audio_guidance_scale (`float`, *optional* defaults to `None`):
                Audio guidance scale for CFG with respect to the negative prompt. The CFG update rule is the same for
                video and audio, but they can use different values for the guidance scale. The LTX-2.X authors suggest
                that the `audio_guidance_scale` should be higher relative to the video `guidance_scale` (e.g. for
                LTX-2.3 they suggest 3.0 for video and 7.0 for audio). If `None`, defaults to the video value
                `guidance_scale`.
            audio_stg_scale (`float`, *optional*, defaults to `None`):
                Audio guidance scale for STG. As with CFG, the STG update rule is otherwise the same for video and
                audio. For LTX-2.3, a value of 1.0 is suggested for both video and audio. If `None`, defaults to the
                video value `stg_scale`.
            audio_modality_scale (`float`, *optional*, defaults to `None`):
                Audio guidance scale for LTX-2.X modality isolation guidance. As with CFG, the modality guidance rule
                is otherwise the same for video and audio. For LTX-2.3, a value of 3.0 is suggested for both video and
                audio. If `None`, defaults to the video value `modality_scale`.
            audio_guidance_rescale (`float`, *optional*, defaults to `None`):
                A separate guidance rescale factor for the audio modality. If `None`, defaults to the video value
                `guidance_rescale`.
            spatio_temporal_guidance_blocks (`list[int]`, *optional*, defaults to `None`):
                The zero-indexed transformer block indices at which to apply STG. Must be supplied if STG is used
                (`stg_scale` or `audio_stg_scale` is greater than `0`). A value of `[29]` is recommended for LTX-2.0
                and `[28]` is recommended for LTX-2.3.
            noise_scale (`float`, *optional*, defaults to `0.0`):
                The interpolation factor between random noise and denoised latents at each timestep. Applying noise to
                the `latents` and `audio_latents` before continue denoising.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            use_cross_timestep (`bool` *optional*, defaults to `False`):
                Whether to use the cross modality (audio is the cross modality of video, and vice versa) sigma when
                calculating the cross attention modulation parameters. `True` is the newer (e.g. LTX-2.3) behavior;
                `False` is the legacy LTX-2.0 behavior.
            system_prompt (`str`, *optional*, defaults to `None`):
                Optional system prompt to use for prompt enhancement. The system prompt will be used by the current
                text encoder (by default, a `Gemma3ForConditionalGeneration` model) to generate an enhanced prompt from
                the original `prompt` to condition generation. If not supplied, prompt enhancement will not be
                performed.
            prompt_max_new_tokens (`int`, *optional*, defaults to `512`):
                The maximum number of new tokens to generate when performing prompt enhancement.
            prompt_enhancement_kwargs (`dict[str, Any]`, *optional*, defaults to `None`):
                Keyword arguments for `self.text_encoder.generate`. If not supplied, default arguments of
                `do_sample=True` and `temperature=0.7` will be used. See
                https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate
                for more details.
            prompt_enhancement_seed (`int`, *optional*, default to `10`):
                Random seed for any random operations during prompt enhancement.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTX2PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to `1024`):
                Maximum sequence length to use with the `prompt`.
            memory_bank ([`JoyAIEchoMemoryBank`], *optional*, defaults to `None`):
                Optional memory bank carrying past shots' pixel-space frames and post-AudioVAE audio latents. When
                `None` or empty, the pipeline behaves identically to the base LTX-2 single-shot path. When non-empty,
                each slot's frames are VAE-encoded on the fly (last latent frame retained per slot, mirroring the
                source recipe at `ltx_distillation/utils.py:102`) and prefix-prepended to the noisy video latents;
                paired audio memory (from `memory_bank.get_memory_audio()`) is prefix-prepended to noisy audio.
                Memory tokens carry σ=0 per-token timesteps (clean), while the target tokens keep the shot's current
                σ. The transformer output is stripped of the memory prefix every denoise step before the scheduler
                update.
            memory_position_mode (`str`, *optional*, defaults to `"legacy"`):
                Position-encoding strategy for the memory prefix. `"legacy"` (the source/inference.yaml default,
                aliased from `"reference"`) keeps memory and target sharing the same frame-0 temporal origin and
                relies on σ=0 + content to disambiguate. `"prefix_continuous"` shifts the target frames so that
                their positional ids continue contiguously after the memory's positional ids. Ignored when
                `memory_bank` is `None` or empty.
            memory_downscale_factor (`int`, *optional*, defaults to `1`):
                Spatial downscale factor applied to memory token H/W coordinates only (temporal coords unaffected),
                useful when memory frames were spatially downscaled. Per `inference.yaml:65`, defaults to `1`.
            paired_audio_memory (`bool`, *optional*, defaults to `True`):
                When `True` (the source default), paired a2v/v2a cross-attention masks AND a memory-block audio
                self-attention mask are enabled in conjunction with the audio memory prefix. C+D.1b only threads
                this flag through; the actual mask construction lands in C+D.1c. With or without this flag, audio
                memory (if present in the bank) is prefix-prepended with sigma=0 per-token timesteps.

        Examples:

        Returns:
            [`~pipelines.ltx.LTX2PipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTX2PipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        audio_guidance_scale = audio_guidance_scale or guidance_scale
        audio_stg_scale = audio_stg_scale or stg_scale
        audio_modality_scale = audio_modality_scale or modality_scale
        audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
        )

        # Per-modality guidance scales (video, audio)
        self._guidance_scale = guidance_scale
        self._stg_scale = stg_scale
        self._modality_scale = modality_scale
        self._guidance_rescale = guidance_rescale
        self._audio_guidance_scale = audio_guidance_scale
        self._audio_stg_scale = audio_stg_scale
        self._audio_modality_scale = audio_modality_scale
        self._audio_guidance_rescale = audio_guidance_rescale

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        if system_prompt is not None and prompt is not None:
            prompt = self.enhance_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=prompt_max_new_tokens,
                seed=prompt_enhancement_seed,
                generator=generator,
                generation_kwargs=prompt_enhancement_kwargs,
                device=device,
            )

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        tokenizer_padding_side = "left"  # Padding side for default Gemma3-12B text encoder
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=tokenizer_padding_side
        )

        # 4. Prepare latent variables
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            if latents.ndim == 5:
                logger.info(
                    "Got latents of shape [batch_size, latent_dim, latent_frames, latent_height, latent_width], `latent_num_frames`, `latent_height`, `latent_width` will be inferred."
                )
                _, _, latent_num_frames, latent_height, latent_width = latents.shape  # [B, C, F, H, W]
            elif latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `latents` of shape {latents.shape}, so the latent dims cannot be"
                    f" inferred. Make sure the supplied `height`, `width`, and `num_frames` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, latent_dim, latent_frames, latent_height, latent_width]."
                )
        # video_sequence_length = latent_num_frames * latent_height * latent_width

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None:
            if audio_latents.ndim == 4:
                logger.info(
                    "Got audio_latents of shape [batch_size, num_channels, audio_length, mel_bins], `audio_num_frames` will be inferred."
                )
                _, _, audio_num_frames, _ = audio_latents.shape  # [B, C, L, M]
            elif audio_latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `audio_latents` of shape {audio_latents.shape}, so the latent dims"
                    f" cannot be inferred. Make sure the supplied `num_frames` and `frame_rate` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `audio_latents` tensor has shape {audio_latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, num_channels, audio_length, mel_bins]."
                )

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )
        audio_latents = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        # For now, duplicate the scheduler for use with the audio latents
        audio_scheduler = copy.deepcopy(self.scheduler)
        _, _ = retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Set begin index to skip nonzero().item() call in scheduler initialization, which triggers GPU sync
        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)

        # 6. Prepare micro-conditions
        # Pre-compute video and audio positional ids as they will be the same at each step of the denoising loop
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        # Duplicate the positional ids as well if using CFG
        if self.do_classifier_free_guidance:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))  # Repeat twice in batch dim
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        # 6b. JoyAI-Echo memory injection prep.
        #
        # When `memory_bank` is None or empty, this entire block is a no-op and the pipeline behaves byte-identically
        # to the LTX-2 base pipeline (preserving the B.4 single-shot parity contract). When the bank carries at least
        # one slot, we:
        #   - VAE-encode each slot's pixel-space frames, keeping only the LAST latent frame per slot (matching the
        #     source recipe at `ltx_distillation/utils.py:102`).
        #   - Pack the resulting memory video latent `[B, C, F_mem, H_lat, W_lat]` into the transformer's
        #     `[B, T_v_mem, C]` flat layout via `_pack_latents`.
        #   - Pull pre-encoded audio memory `[B, T_a_mem, C]` directly off the bank (the bank stores audio in the
        #     packed normalized latent space; no further VAE call required).
        #   - Pre-extend `video_coords` / `audio_coords` to cover the memory prefix (MEMORY FIRST), following
        #     `memory_position_mode`: `legacy` (the default, aliased from the source's `reference` setting) shares
        #     frame-0 origin between memory and target; `prefix_continuous` shifts target positions by the memory
        #     length so the two prefixes occupy contiguous position ranges.
        # The σ=0 per-token timesteps and the prefix-strip on output are applied PER transformer call inside the
        # denoise loop (every step), so memory tokens never enter the scheduler.step state for the target latents.
        memory_video_seq_len = 0
        memory_audio_seq_len = 0
        memory_video_flat: torch.Tensor | None = None
        memory_audio_flat: torch.Tensor | None = None

        memory_position_mode_normalized = str(memory_position_mode).lower()
        if memory_position_mode_normalized == "reference":
            memory_position_mode_normalized = "legacy"
        if memory_position_mode_normalized not in {"legacy", "prefix_continuous"}:
            raise ValueError(
                "memory_position_mode must be one of {'legacy', 'reference', 'prefix_continuous'}, "
                f"got {memory_position_mode}"
            )

        if memory_bank is not None and len(memory_bank) > 0:
            # ---- Encode pixel-space memory frames once per pipeline call ----
            memory_dtype = prompt_embeds.dtype
            slot_frames_list = memory_bank.get_memory_frames()
            num_memory_slots = len(slot_frames_list)
            # The VAE encode below stochastically samples from the latent distribution; we MUST thread the
            # caller's `generator` through so that two same-seed calls with the same memory bank produce
            # bit-identical outputs (C+D.2b P0 fix). `DiagonalGaussianDistribution.sample` accepts a
            # `torch.Generator | None`. The pipeline `generator` may be a single Generator, a list of
            # per-batch Generators, or None. The slot batch dim is always 1 (we encode one slot at a time),
            # so we pick the first entry when a list is passed; for None we pass through.
            if isinstance(generator, list):
                vae_encode_generator = generator[0] if len(generator) > 0 else None
            else:
                vae_encode_generator = generator
            per_slot_latents: list[torch.Tensor] = []
            for slot_idx, slot_frames in enumerate(slot_frames_list):
                if not slot_frames:
                    raise ValueError(
                        f"`memory_bank` slot {slot_idx} has no frames; cannot VAE-encode an empty memory clip."
                    )
                # Convert PIL frames -> [1, 3, F_pixel, H, W] in [-1, 1] (matches the source convention at
                # `ltx_distillation/utils.py:frames_to_video_tensor`). Required height/width are set by the
                # pipeline `height`/`width` arguments above; the bank does not validate frame size, so we
                # enforce here that all slot frames match the requested resolution.
                frame_tensors = []
                for frame_idx, frame in enumerate(slot_frames):
                    if frame.size != (width, height):
                        raise ValueError(
                            f"memory_bank slot {slot_idx} frame {frame_idx} has size {frame.size}, expected "
                            f"({width}, {height}) (pipeline `width`, `height`)."
                        )
                    frame_tensor = TVF.to_tensor(frame) * 2.0 - 1.0  # [3, H, W] in [-1, 1]
                    frame_tensors.append(frame_tensor)
                pixel_clip = torch.stack(frame_tensors, dim=1).unsqueeze(0)  # [1, 3, F_pixel, H, W]
                pixel_clip = pixel_clip.to(device=device, dtype=self.vae.dtype)
                latent_dist = self.vae.encode(pixel_clip).latent_dist
                slot_latent = latent_dist.sample(generator=vae_encode_generator)  # [1, C, F_lat, H_lat, W_lat]
                # Keep only the LAST latent frame (one latent frame per slot, mirroring the source recipe at
                # `ltx_distillation/utils.py:102`).
                slot_latent = slot_latent[:, :, -1:, :, :].contiguous()
                per_slot_latents.append(slot_latent)
            # Concat over the latent-frame dim -> [1, C, F_mem, H_lat, W_lat].
            memory_video_latent = torch.cat(per_slot_latents, dim=2)
            # Normalize using the same statistics as the noisy target latents (the target is normalized in
            # `prepare_latents` via `_normalize_latents`).
            memory_video_latent = self._normalize_latents(
                memory_video_latent, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            # Pack to flat token sequence [1, T_v_mem, C] and broadcast to the actual per-call batch.
            memory_video_flat = self._pack_latents(
                memory_video_latent,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            if memory_video_flat.shape[0] == 1 and latents.shape[0] != 1:
                memory_video_flat = memory_video_flat.expand(latents.shape[0], -1, -1).contiguous()
            memory_video_flat = memory_video_flat.to(device=device, dtype=memory_dtype)
            memory_video_seq_len = memory_video_flat.shape[1]
            num_memory_latent_frames = memory_video_latent.shape[2]

            # ---- Extend video_coords with the memory prefix ----
            # In `legacy` mode, both memory and target start at frame 0 (sharing temporal origin); the model
            # disambiguates via σ=0 and content. In `prefix_continuous` mode, target frames are shifted by the
            # number of memory latent frames so the two prefixes occupy contiguous position ranges.
            memory_video_coords = self.transformer.rope.prepare_video_coords(
                latents.shape[0],
                num_memory_latent_frames,
                latent_height,
                latent_width,
                device,
                fps=frame_rate,
            )
            if memory_downscale_factor != 1:
                memory_video_coords = memory_video_coords.clone()
                memory_video_coords[:, 1, ...] *= float(memory_downscale_factor)
                memory_video_coords[:, 2, ...] *= float(memory_downscale_factor)
            if memory_position_mode_normalized == "prefix_continuous":
                target_frame_shift_seconds = float(num_memory_latent_frames) * (
                    self.vae_temporal_compression_ratio / float(frame_rate)
                )
                video_coords = video_coords.clone()
                video_coords[:, 0, ...] = video_coords[:, 0, ...] + target_frame_shift_seconds
            if self.do_classifier_free_guidance:
                memory_video_coords = memory_video_coords.repeat((2,) + (1,) * (memory_video_coords.ndim - 1))
            video_coords = torch.cat([memory_video_coords, video_coords], dim=2)

            # ---- Audio memory (only when the bank carries audio latents for every slot) ----
            memory_audio_latent = memory_bank.get_memory_audio()
            if memory_audio_latent is not None:
                memory_audio_flat = memory_audio_latent.to(device=device, dtype=memory_dtype)
                if memory_audio_flat.shape[0] == 1 and audio_latents.shape[0] != 1:
                    memory_audio_flat = memory_audio_flat.expand(audio_latents.shape[0], -1, -1).contiguous()
                memory_audio_seq_len = memory_audio_flat.shape[1]

                memory_audio_coords = self.transformer.audio_rope.prepare_audio_coords(
                    audio_latents.shape[0], memory_audio_seq_len, device
                )
                if memory_position_mode_normalized == "prefix_continuous":
                    target_audio_shift_seconds = float(memory_audio_seq_len) * (
                        self.audio_hop_length
                        * float(self.audio_vae_temporal_compression_ratio)
                        / float(self.audio_sampling_rate)
                    )
                    audio_coords = audio_coords.clone()
                    audio_coords[:, 0, ...] = audio_coords[:, 0, ...] + target_audio_shift_seconds
                if self.do_classifier_free_guidance:
                    memory_audio_coords = memory_audio_coords.repeat((2,) + (1,) * (memory_audio_coords.ndim - 1))
                audio_coords = torch.cat([memory_audio_coords, audio_coords], dim=2)

            del per_slot_latents

        has_memory = memory_video_seq_len > 0 or memory_audio_seq_len > 0

        # ---- JoyAI-Echo paired-memory attention masks (C+D.1c) ----
        # Built once per pipeline call at base batch size B (`latents.shape[0]`); the per-call-site
        # batch fan-out below mirrors the `_prepend_memory_*` helpers' block-tiling (`.repeat`, NOT
        # `repeat_interleave`, to match the CFG `cat([x]*2, dim=0)` layout).
        #
        # Gating (source `ltx_wrapper.py:649`): all three masks are populated ONLY when
        # `paired_audio_memory=True`, video memory is non-empty, AND audio memory is non-empty. When
        # `paired_audio_memory=False`, the masks stay None — audio falls back to full attention and
        # the cross-attns run unmasked (C+D.1b's memory injection still applies via the σ=0 prefix).
        audio_self_mask_base: torch.Tensor | None = None
        a2v_kwarg_mask_base: torch.Tensor | None = None  # SWAPPED: source's v2a_pairwise_mask
        v2a_kwarg_mask_base: torch.Tensor | None = None  # SWAPPED: source's a2v_pairwise_mask
        if (
            bool(paired_audio_memory)
            and memory_video_seq_len > 0
            and memory_audio_seq_len > 0
            and memory_bank is not None
        ):
            num_memory_slots = len(memory_bank)
            memory_audio_segment_lengths = memory_bank.get_memory_audio_segment_lengths() or None
            base_batch_size = latents.shape[0]
            target_video_seq_len_base = latents.shape[1]
            target_audio_seq_len_base = audio_latents.shape[1]
            mask_dtype = prompt_embeds.dtype

            audio_self_mask_base = self._build_audio_self_attention_mask(
                batch_size=base_batch_size,
                memory_seq_len=memory_audio_seq_len,
                target_seq_len=target_audio_seq_len_base,
                device=device,
                dtype=mask_dtype,
            )
            # `a2v_pairwise_mask` follows the data-flow naming: Q=audio (T_a), KV=video (T_v);
            # shape (B, T_a, T_v). NOTE:the source-code variable of the same name has the OPPOSITE
            # convention (Q=video); only the data-flow naming used in our memory file matches this
            # local. The final wiring below (L1587) handles the source↔diffusers kwarg SWAP.
            # Audio dim uses real per-slot lengths; video dim uses equal-division fallback.
            a2v_pairwise_mask = self._build_paired_memory_cross_mask(
                batch_size=base_batch_size,
                query_memory_seq_len=memory_audio_seq_len,
                query_target_seq_len=target_audio_seq_len_base,
                kv_memory_seq_len=memory_video_seq_len,
                kv_target_seq_len=target_video_seq_len_base,
                num_memory_slots=num_memory_slots,
                device=device,
                dtype=mask_dtype,
                query_segment_lengths=memory_audio_segment_lengths,
                kv_segment_lengths=None,
            )
            # `v2a_pairwise_mask` follows the data-flow naming: Q=video (T_v), KV=audio (T_a);
            # shape (B, T_v, T_a). Same convention caveat as the `a2v_pairwise_mask` comment above.
            v2a_pairwise_mask = self._build_paired_memory_cross_mask(
                batch_size=base_batch_size,
                query_memory_seq_len=memory_video_seq_len,
                query_target_seq_len=target_video_seq_len_base,
                kv_memory_seq_len=memory_audio_seq_len,
                kv_target_seq_len=target_audio_seq_len_base,
                num_memory_slots=num_memory_slots,
                device=device,
                dtype=mask_dtype,
                query_segment_lengths=None,
                kv_segment_lengths=memory_audio_segment_lengths,
            )
            # ---- Source ↔ diffusers kwarg SWAP (see memory: feedback-a2v-v2a-naming-swap) ----
            # Source picks "data-flow direction" (a2v = audio is source); diffusers picks "layer
            # purpose" (a2v_cross_attention_mask is the mask FOR the a2v attn layer, where Q=video,
            # KV=audio). Same shapes — only the kwarg NAMES cross. No .transpose() needed.
            a2v_kwarg_mask_base = v2a_pairwise_mask  # diffusers' a2v takes source's v2a (Q=video)
            v2a_kwarg_mask_base = a2v_pairwise_mask  # diffusers' v2a takes source's a2v (Q=audio)

        def _fan_out_mask(mask: torch.Tensor | None, target_batch: int) -> torch.Tensor | None:
            """Block-tile a `[B, ...]` mask to match a CFG-doubled `[target_batch, ...]` layout.

            Mirrors `_prepend_memory_*` batch-tiling: uses `.repeat` (NOT
            `repeat_interleave`) so the layout matches the pipeline's CFG `cat([x]*2, dim=0)`
            convention (block-replicated: `[uncond_B | cond_B]`). C+D.2b P2 rationale applies.
            """
            if mask is None:
                return None
            if mask.shape[0] == target_batch:
                return mask
            if mask.shape[0] < target_batch:
                repeat_factor = target_batch // mask.shape[0]
                # repeat along batch dim only; keep all other dims unchanged
                expand_args = (repeat_factor,) + (1,) * (mask.ndim - 1)
                mask = mask.repeat(*expand_args)
            if mask.shape[0] > target_batch:
                mask = mask[:target_batch]
            return mask

        def _prepend_memory_video(target_video: torch.Tensor) -> torch.Tensor:
            """Prepend the cached memory video prefix to a target latent of shape `[B_eff, T_v_tgt, C]`."""
            if memory_video_flat is None:
                return target_video
            mem = memory_video_flat
            if mem.shape[0] != target_video.shape[0]:
                # CFG / STG branches may pass `target` with batch == B (no CFG doubling) even when the cached
                # memory was sized to 2B (or vice versa). Tile / trim along the batch dim. The CFG layout is
                # `cat([x]*2, dim=0)` (block-replicated: `[uncond_B | cond_B]`) so we MUST block-tile the
                # memory the same way — `repeat_interleave` would interleave rows and misalign per-row memory
                # for B>1 (C+D.2b P2 fix). `expand` would only work for singleton dims.
                if mem.shape[0] < target_video.shape[0]:
                    repeat_factor = target_video.shape[0] // mem.shape[0]
                    mem = mem.repeat(repeat_factor, 1, 1)
                if mem.shape[0] > target_video.shape[0]:
                    mem = mem[: target_video.shape[0]]
            mem = mem.to(dtype=target_video.dtype)
            return torch.cat([mem, target_video], dim=1)

        def _prepend_memory_audio(target_audio: torch.Tensor) -> torch.Tensor:
            """Prepend the cached memory audio prefix to a target latent of shape `[B_eff, T_a_tgt, C]`."""
            if memory_audio_flat is None:
                return target_audio
            mem = memory_audio_flat
            if mem.shape[0] != target_audio.shape[0]:
                # Block-tile to match the CFG `cat([x]*2, dim=0)` layout (C+D.2b P2 fix); see
                # `_prepend_memory_video` for rationale.
                if mem.shape[0] < target_audio.shape[0]:
                    repeat_factor = target_audio.shape[0] // mem.shape[0]
                    mem = mem.repeat(repeat_factor, 1, 1)
                if mem.shape[0] > target_audio.shape[0]:
                    mem = mem[: target_audio.shape[0]]
            mem = mem.to(dtype=target_audio.dtype)
            return torch.cat([mem, target_audio], dim=1)

        def _per_token_timestep_with_memory_prefix(
            scalar_timestep: torch.Tensor, target_seq_len: int, memory_seq_len: int
        ) -> torch.Tensor:
            """Build a per-token timestep of shape `[B_eff, memory_seq_len + target_seq_len]`.

            The memory prefix carries σ=0 (clean) per the source recipe
            (`ltx_distillation/models/ltx_wrapper.py:536-542`); the target tokens carry the shot's current σ. This
            matches the transformer's documented per-token timestep contract at
            `transformer_ltx2_joyai_echo.py:350-352`.
            """
            B_eff = scalar_timestep.shape[0]
            target_timestep = scalar_timestep.view(B_eff, 1).expand(B_eff, target_seq_len)
            if memory_seq_len <= 0:
                return target_timestep
            memory_timestep = torch.zeros(
                B_eff, memory_seq_len, device=scalar_timestep.device, dtype=scalar_timestep.dtype
            )
            return torch.cat([memory_timestep, target_timestep], dim=1)

        def _strip_memory_prefix_video(pred: torch.Tensor) -> torch.Tensor:
            if memory_video_seq_len <= 0:
                return pred
            return pred[:, memory_video_seq_len:, :]

        def _strip_memory_prefix_audio(pred: torch.Tensor) -> torch.Tensor:
            if memory_audio_seq_len <= 0:
                return pred
            return pred[:, memory_audio_seq_len:, :]

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if has_memory:
                    target_video_seq_len_0 = latent_model_input.shape[1]
                    target_audio_seq_len_0 = audio_latent_model_input.shape[1]
                    latent_model_input_with_mem = _prepend_memory_video(latent_model_input)
                    audio_latent_model_input_with_mem = _prepend_memory_audio(audio_latent_model_input)
                    video_timestep_for_call = _per_token_timestep_with_memory_prefix(
                        timestep, target_video_seq_len_0, memory_video_seq_len
                    )
                    audio_timestep_for_call = _per_token_timestep_with_memory_prefix(
                        timestep, target_audio_seq_len_0, memory_audio_seq_len
                    )
                    # Cross-attn AdaLN must use the SCALAR target σ (per-row, no σ=0 prefix), matching the
                    # source `Modality.sigma` semantics. The per-token σ=0 prefix lives on the self-attn
                    # AdaLN path (`video_timestep_for_call` / `audio_timestep_for_call`). C+D.2b P1 fix.
                    video_cross_attention_sigma_for_call = timestep
                    audio_cross_attention_sigma_for_call = timestep
                    # Block-tile the paired-memory masks (built at base B) to match the CFG-doubled
                    # batch of this transformer call. C+D.1c uses the same `.repeat` (block-replicated)
                    # convention as `_prepend_memory_*` to stay consistent with the `cat([x]*2, dim=0)`
                    # CFG layout. Masks are None when `paired_audio_memory=False`.
                    cu_batch = latent_model_input_with_mem.shape[0]
                    audio_self_mask_call = _fan_out_mask(audio_self_mask_base, cu_batch)
                    a2v_kwarg_mask_call = _fan_out_mask(a2v_kwarg_mask_base, cu_batch)
                    v2a_kwarg_mask_call = _fan_out_mask(v2a_kwarg_mask_base, cu_batch)
                else:
                    latent_model_input_with_mem = latent_model_input
                    audio_latent_model_input_with_mem = audio_latent_model_input
                    video_timestep_for_call = timestep
                    audio_timestep_for_call = None
                    video_cross_attention_sigma_for_call = None
                    audio_cross_attention_sigma_for_call = None
                    audio_self_mask_call = None
                    a2v_kwarg_mask_call = None
                    v2a_kwarg_mask_call = None

                with self.transformer.cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio = self.transformer(
                        hidden_states=latent_model_input_with_mem,
                        audio_hidden_states=audio_latent_model_input_with_mem,
                        encoder_hidden_states=connector_prompt_embeds,
                        audio_encoder_hidden_states=connector_audio_prompt_embeds,
                        timestep=video_timestep_for_call,
                        audio_timestep=audio_timestep_for_call,
                        sigma=timestep,  # Used by LTX-2.3 (scalar per-batch sigma; the sigma=0 prefix lives on `timestep`)
                        encoder_attention_mask=connector_attention_mask,
                        audio_encoder_attention_mask=connector_attention_mask,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=frame_rate,
                        audio_num_frames=audio_num_frames,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        isolate_modalities=False,
                        spatio_temporal_guidance_blocks=None,
                        perturbation_mask=None,
                        use_cross_timestep=use_cross_timestep,
                        attention_kwargs=attention_kwargs,
                        video_cross_attention_sigma=video_cross_attention_sigma_for_call,
                        audio_cross_attention_sigma=audio_cross_attention_sigma_for_call,
                        audio_self_attention_mask=audio_self_mask_call,
                        a2v_cross_attention_mask=a2v_kwarg_mask_call,
                        v2a_cross_attention_mask=v2a_kwarg_mask_call,
                        return_dict=False,
                    )
                noise_pred_video = _strip_memory_prefix_video(noise_pred_video)
                noise_pred_audio = _strip_memory_prefix_audio(noise_pred_audio)
                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond_text, noise_pred_video = noise_pred_video.chunk(2)
                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_video_uncond_text = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_text, i, self.scheduler
                    )
                    # Use delta formulation as it works more nicely with multiple guidance terms
                    video_cfg_delta = (self.guidance_scale - 1) * (noise_pred_video - noise_pred_video_uncond_text)

                    noise_pred_audio_uncond_text, noise_pred_audio = noise_pred_audio.chunk(2)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)
                    noise_pred_audio_uncond_text = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_text, i, audio_scheduler
                    )
                    audio_cfg_delta = (self.audio_guidance_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_text
                    )

                    # Get positive values from merged CFG inputs in case we need to do other DiT forward passes
                    if self.do_spatio_temporal_guidance or self.do_modality_isolation_guidance:
                        if i == 0:
                            # Only split values that remain constant throughout the loop once
                            video_prompt_embeds = connector_prompt_embeds.chunk(2, dim=0)[1]
                            audio_prompt_embeds = connector_audio_prompt_embeds.chunk(2, dim=0)[1]
                            prompt_attn_mask = connector_attention_mask.chunk(2, dim=0)[1]

                            video_pos_ids = video_coords.chunk(2, dim=0)[0]
                            audio_pos_ids = audio_coords.chunk(2, dim=0)[0]

                        # Split values that vary each denoising loop iteration
                        timestep = timestep.chunk(2, dim=0)[0]
                else:
                    video_cfg_delta = audio_cfg_delta = 0

                    video_prompt_embeds = connector_prompt_embeds
                    audio_prompt_embeds = connector_audio_prompt_embeds
                    prompt_attn_mask = connector_attention_mask

                    video_pos_ids = video_coords
                    audio_pos_ids = audio_coords

                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)

                if self.do_spatio_temporal_guidance:
                    if has_memory:
                        stg_video_input = _prepend_memory_video(latents.to(dtype=prompt_embeds.dtype))
                        stg_audio_input = _prepend_memory_audio(audio_latents.to(dtype=prompt_embeds.dtype))
                        stg_video_timestep = _per_token_timestep_with_memory_prefix(
                            timestep, latents.shape[1], memory_video_seq_len
                        )
                        stg_audio_timestep = _per_token_timestep_with_memory_prefix(
                            timestep, audio_latents.shape[1], memory_audio_seq_len
                        )
                        # Scalar target σ for cross-attn AdaLN (C+D.2b P1 fix); see cond_uncond branch.
                        stg_video_cross_attention_sigma = timestep
                        stg_audio_cross_attention_sigma = timestep
                        # Block-tile paired-memory masks to STG branch batch (typically base B, since
                        # STG runs the *positive* branch only — no CFG doubling here).
                        stg_batch = stg_video_input.shape[0]
                        stg_audio_self_mask = _fan_out_mask(audio_self_mask_base, stg_batch)
                        stg_a2v_kwarg_mask = _fan_out_mask(a2v_kwarg_mask_base, stg_batch)
                        stg_v2a_kwarg_mask = _fan_out_mask(v2a_kwarg_mask_base, stg_batch)
                    else:
                        stg_video_input = latents.to(dtype=prompt_embeds.dtype)
                        stg_audio_input = audio_latents.to(dtype=prompt_embeds.dtype)
                        stg_video_timestep = timestep
                        stg_audio_timestep = None
                        stg_video_cross_attention_sigma = None
                        stg_audio_cross_attention_sigma = None
                        stg_audio_self_mask = None
                        stg_a2v_kwarg_mask = None
                        stg_v2a_kwarg_mask = None

                    with self.transformer.cache_context("uncond_stg"):
                        noise_pred_video_uncond_stg, noise_pred_audio_uncond_stg = self.transformer(
                            hidden_states=stg_video_input,
                            audio_hidden_states=stg_audio_input,
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=stg_video_timestep,
                            audio_timestep=stg_audio_timestep,
                            sigma=timestep,  # Used by LTX-2.3 (scalar per-batch sigma; sigma=0 prefix lives on `timestep`)
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            isolate_modalities=False,
                            # Use STG at given blocks to perturb model
                            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            video_cross_attention_sigma=stg_video_cross_attention_sigma,
                            audio_cross_attention_sigma=stg_audio_cross_attention_sigma,
                            audio_self_attention_mask=stg_audio_self_mask,
                            a2v_cross_attention_mask=stg_a2v_kwarg_mask,
                            v2a_cross_attention_mask=stg_v2a_kwarg_mask,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_stg = _strip_memory_prefix_video(noise_pred_video_uncond_stg)
                    noise_pred_audio_uncond_stg = _strip_memory_prefix_audio(noise_pred_audio_uncond_stg)
                    noise_pred_video_uncond_stg = noise_pred_video_uncond_stg.float()
                    noise_pred_audio_uncond_stg = noise_pred_audio_uncond_stg.float()
                    noise_pred_video_uncond_stg = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_stg, i, self.scheduler
                    )
                    noise_pred_audio_uncond_stg = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_stg, i, audio_scheduler
                    )

                    video_stg_delta = self.stg_scale * (noise_pred_video - noise_pred_video_uncond_stg)
                    audio_stg_delta = self.audio_stg_scale * (noise_pred_audio - noise_pred_audio_uncond_stg)
                else:
                    video_stg_delta = audio_stg_delta = 0

                if self.do_modality_isolation_guidance:
                    if has_memory:
                        mod_video_input = _prepend_memory_video(latents.to(dtype=prompt_embeds.dtype))
                        mod_audio_input = _prepend_memory_audio(audio_latents.to(dtype=prompt_embeds.dtype))
                        mod_video_timestep = _per_token_timestep_with_memory_prefix(
                            timestep, latents.shape[1], memory_video_seq_len
                        )
                        mod_audio_timestep = _per_token_timestep_with_memory_prefix(
                            timestep, audio_latents.shape[1], memory_audio_seq_len
                        )
                        # Scalar target σ for cross-attn AdaLN (C+D.2b P1 fix); see cond_uncond branch.
                        mod_video_cross_attention_sigma = timestep
                        mod_audio_cross_attention_sigma = timestep
                        # Block-tile paired-memory masks to mod-iso branch batch.
                        mod_batch = mod_video_input.shape[0]
                        mod_audio_self_mask = _fan_out_mask(audio_self_mask_base, mod_batch)
                        mod_a2v_kwarg_mask = _fan_out_mask(a2v_kwarg_mask_base, mod_batch)
                        mod_v2a_kwarg_mask = _fan_out_mask(v2a_kwarg_mask_base, mod_batch)
                    else:
                        mod_video_input = latents.to(dtype=prompt_embeds.dtype)
                        mod_audio_input = audio_latents.to(dtype=prompt_embeds.dtype)
                        mod_video_timestep = timestep
                        mod_audio_timestep = None
                        mod_video_cross_attention_sigma = None
                        mod_audio_cross_attention_sigma = None
                        mod_audio_self_mask = None
                        mod_a2v_kwarg_mask = None
                        mod_v2a_kwarg_mask = None

                    with self.transformer.cache_context("uncond_modality"):
                        noise_pred_video_uncond_modality, noise_pred_audio_uncond_modality = self.transformer(
                            hidden_states=mod_video_input,
                            audio_hidden_states=mod_audio_input,
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=mod_video_timestep,
                            audio_timestep=mod_audio_timestep,
                            sigma=timestep,  # Used by LTX-2.3 (scalar per-batch sigma; sigma=0 prefix lives on `timestep`)
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            # Turn off A2V and V2A cross attn to isolate video and audio modalities
                            isolate_modalities=True,
                            spatio_temporal_guidance_blocks=None,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            video_cross_attention_sigma=mod_video_cross_attention_sigma,
                            audio_cross_attention_sigma=mod_audio_cross_attention_sigma,
                            audio_self_attention_mask=mod_audio_self_mask,
                            a2v_cross_attention_mask=mod_a2v_kwarg_mask,
                            v2a_cross_attention_mask=mod_v2a_kwarg_mask,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_modality = _strip_memory_prefix_video(noise_pred_video_uncond_modality)
                    noise_pred_audio_uncond_modality = _strip_memory_prefix_audio(noise_pred_audio_uncond_modality)
                    noise_pred_video_uncond_modality = noise_pred_video_uncond_modality.float()
                    noise_pred_audio_uncond_modality = noise_pred_audio_uncond_modality.float()
                    noise_pred_video_uncond_modality = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_modality, i, self.scheduler
                    )
                    noise_pred_audio_uncond_modality = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_modality, i, audio_scheduler
                    )

                    video_modality_delta = (self.modality_scale - 1) * (
                        noise_pred_video - noise_pred_video_uncond_modality
                    )
                    audio_modality_delta = (self.audio_modality_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_modality
                    )
                else:
                    video_modality_delta = audio_modality_delta = 0

                # Now apply all guidance terms
                noise_pred_video_g = noise_pred_video + video_cfg_delta + video_stg_delta + video_modality_delta
                noise_pred_audio_g = noise_pred_audio + audio_cfg_delta + audio_stg_delta + audio_modality_delta

                # Apply LTX-2.X guidance rescaling
                if self.guidance_rescale > 0:
                    noise_pred_video = rescale_noise_cfg(
                        noise_pred_video_g, noise_pred_video, guidance_rescale=self.guidance_rescale
                    )
                else:
                    noise_pred_video = noise_pred_video_g

                if self.audio_guidance_rescale > 0:
                    noise_pred_audio = rescale_noise_cfg(
                        noise_pred_audio_g, noise_pred_audio, guidance_rescale=self.audio_guidance_rescale
                    )
                else:
                    noise_pred_audio = noise_pred_audio_g

                # Convert back to velocity for scheduler
                noise_pred_video = self.convert_x0_to_velocity(latents, noise_pred_video, i, self.scheduler)
                noise_pred_audio = self.convert_x0_to_velocity(audio_latents, noise_pred_audio, i, audio_scheduler)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                # NOTE: for now duplicate scheduler for audio latents in case self.scheduler sets internal state in
                # the step method (such as _step_index)
                audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        # Snapshot the packed normalized audio latent for the optional memory-bank push at the end of
        # the call. The bank consumers (C+D.1b: `memory_audio_flat` at line 1560) read this same
        # packed-normalized 3-D `[B, T_a, C]` form, so we must capture BEFORE the
        # `_denormalize_audio_latents` + `_unpack_audio_latents` calls below transform it to 4-D
        # `[B, C, L, M]` denormalized.
        audio_latent_for_bank: Optional[torch.Tensor] = (
            audio_latents.detach() if (paired_audio_memory and memory_bank is not None) else None
        )

        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if output_type == "latent":
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
            audio = audio_latents
        else:
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )

            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            # Keep the raw decoded video tensor for the memory-bank push (which needs PIL frames)
            # before user-facing postprocessing potentially converts to np/pt.
            video_for_bank = video if (paired_audio_memory and memory_bank is not None) else None
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(generated_mel_spectrograms)

            # ---- C+D.1d: end-of-shot push into the paired memory bank ----
            # Run only when the caller opted into paired audio memory AND provided a bank. This
            # mirrors the source's per-shot save (`memory_multishot.py`'s wrapper-class loop end:
            # `save_memory_slot(...)`) but lives inline here so multi-shot inference auto-populates
            # the bank without the caller having to round-trip through their own loop.
            if paired_audio_memory and memory_bank is not None and audio_latent_for_bank is not None:
                # 1) PIL frames for the first batch element of the just-generated clip.
                pil_video_batches = self.video_processor.postprocess_video(video_for_bank, output_type="pil")
                pil_video = list(pil_video_batches[0])

                # 2) Mel spectrogram over the decoded waveform (first batch element). The vocoder
                # returns `[B, T]` or `[B, C, T]`; either is handled by `_compute_mel_for_audio_selection`.
                waveform_for_mel = audio[0] if audio.dim() >= 2 else audio
                mel = self._compute_mel_for_audio_selection(
                    waveform_for_mel,
                    sample_rate=int(audio_memory_sample_rate),
                    n_fft=int(audio_memory_n_fft),
                    hop_length=int(audio_memory_mel_hop_length),
                    mel_bins=int(audio_memory_mel_bins),
                )

                # 3) Pick the audio window (returns latent + mel half-open bounds).
                lat_start, lat_end, mel_start, mel_end = self._select_audio_window_with_bounds(
                    mel,
                    window_size_latent=int(audio_memory_window_size),
                    downsample_factor=int(audio_memory_downsample_factor),
                    selection_mode=audio_memory_window_selection_mode,
                    is_causal=bool(audio_memory_is_causal),
                    generator=generator if isinstance(generator, torch.Generator) else None,
                )

                # 4) Slice the packed-normalized audio latent on the time axis.
                audio_total_T = audio_latent_for_bank.shape[1]
                clamped_start = max(0, min(int(lat_start), audio_total_T))
                clamped_end = max(clamped_start, min(int(lat_end), audio_total_T))
                if clamped_end == clamped_start:
                    clamped_end = min(audio_total_T, clamped_start + 1)
                # Keep only the first batch element so the bank stores `[1, T_a_slot, C]` regardless
                # of caller batch (multi-sample memory is not currently supported by the bank API).
                audio_slot = audio_latent_for_bank[:1, clamped_start:clamped_end, :].detach().cpu().contiguous()

                # 5) Map window to time, pick video frame indices, and slice PIL frames.
                t_start, t_end = self._mel_window_bounds_to_seconds(
                    mel_start,
                    mel_end,
                    hop_length=int(audio_memory_mel_hop_length),
                    sample_rate=int(audio_memory_sample_rate),
                )
                frame_indices = self._select_video_frame_indices_from_time_range(
                    t_start=float(t_start),
                    t_end=float(t_end),
                    video_fps=float(frame_rate),
                    total_video_frames=len(pil_video),
                    clip_num_frames=int(video_memory_clip_num_frames),
                )
                pil_clip = [pil_video[i] for i in frame_indices]

                memory_bank.save_memory_slot(
                    frames=pil_clip,
                    audio_latent=audio_slot,
                    metadata={
                        "shot_audio_window_seconds": (float(t_start), float(t_end)),
                        "shot_audio_latent_window": (int(clamped_start), int(clamped_end)),
                        "shot_audio_mel_window": (int(mel_start), int(mel_end)),
                        "shot_video_clip_indices": list(frame_indices),
                        "shot_audio_window_selection_mode": str(audio_memory_window_selection_mode).lower(),
                    },
                )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)

        return LTX2PipelineOutput(frames=video, audio=audio)
