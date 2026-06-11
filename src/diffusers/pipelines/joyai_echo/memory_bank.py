# Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
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
"""Memory bank data structure for the JoyAI-Echo multi-shot pipeline.

This module provides the in-memory ring-buffer used by ``JoyAIEchoPipeline`` to
carry pixel-space video frames and (post-AudioVAE) audio latents from past
shots into the current shot's denoise call.

Pure data structure -- no model, no VAE, no mask logic. PIL <-> tensor
conversion is the pipeline's responsibility (see ``C+D.1b``); paired
cross-attention masks live next to the transformer (see ``C+D.1c``). The bank
API is intentionally dumb: the caller hands it pre-selected frames + audio
latent, and the bank only enforces the storage contract (CPU, contiguous,
detached, shape-checked) plus the FIFO-with-sticky-prefix eviction policy.
"""

# Adapted from JoyAI-Echo's PairedAudioVideoMemoryBank
# (ltx_distillation/inference/memory_multishot.py).
# The source class also owns audio-window / video-clip selection helpers; here
# those concerns live in the pipeline wrapper instead, keeping the bank a
# minimal data container.

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import PIL.Image
import torch


@dataclass
class MemoryEntry:
    """One memory slot: a short pixel-space video clip + its paired audio latent.

    Frames are kept as PIL Images (host memory) and are NOT pre-encoded into VAE
    latents -- the pipeline VAE-encodes them every shot at use-time, mirroring
    the "store cold, encode hot" pattern of the source pipeline.

    Args:
        frames: The per-slot pixel-space clip (typically 9 frames in the source
            recipe). Must be a non-empty ``list`` of :class:`PIL.Image.Image`.
        audio_latent: Optional post-AudioVAE / post-patchify audio latent of
            shape ``[B, T_a_slot, C]`` on CPU (detached, contiguous). May be
            ``None`` for forward-compatible video-only memory; in that case
            this slot is skipped when concatenating audio memory.
        metadata: Free-form per-slot metadata. The pipeline populates this with
            clip-alignment info (e.g. selected audio-window bounds, video-clip
            center frame); the bank itself never reads it.
    """

    frames: list[PIL.Image.Image]
    audio_latent: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JoyAIEchoMemoryBank:
    """FIFO memory bank with a sticky prefix of "fixed" early shots.

    The bank is the diffusers-side equivalent of JoyAI-Echo's
    ``PairedAudioVideoMemoryBank``. It stores up to ``max_size`` slots; the
    first ``num_fix_frames`` slots saved are sticky (never evicted), and the
    remaining capacity acts as a FIFO over later shots. This matches the source
    eviction rule (see :meth:`_trim`).

    The bank is deliberately model-free: it does no VAE encoding, no PIL <->
    tensor conversion, no audio-window selection, and no attention mask
    construction. Those concerns belong to the pipeline (selection /
    conversion) and the transformer call site (masks).

    Args:
        max_size: Maximum number of slots retained (default ``7``, matching
            ``inference.yaml: memory.max_size``).
        num_fix_frames: Number of leading slots treated as sticky -- never
            evicted regardless of bank growth (default ``3``, matching
            ``inference.yaml: memory.num_fix_frames`` -- note: differs from
            the source class default of ``0``). Must satisfy
            ``0 <= num_fix_frames <= max_size``.
        save_mode: Forward-looking knob from the source API. Only ``"frames"``
            is currently meaningful (pixel-space PIL storage). Retained as a
            kwarg for future modes (e.g. pre-encoded latent caching).

    Example:
        >>> bank = JoyAIEchoMemoryBank(max_size=4, num_fix_frames=2)
        >>> bank.save_memory_slot(frames=[img0, ...], audio_latent=lat0)  # doctest: +SKIP
        >>> len(bank)
        1
    """

    def __init__(
        self,
        max_size: int = 7,
        num_fix_frames: int = 3,
        save_mode: str = "frames",
    ) -> None:
        max_size = int(max_size)
        num_fix_frames = int(num_fix_frames)
        if max_size < 0:
            raise ValueError(f"`max_size` must be >= 0, got {max_size}.")
        if not (0 <= num_fix_frames <= max_size):
            raise ValueError(
                "`num_fix_frames` must satisfy 0 <= num_fix_frames <= max_size, "
                f"got num_fix_frames={num_fix_frames}, max_size={max_size}."
            )
        if save_mode != "frames":
            raise ValueError(f"Only save_mode='frames' is currently supported, got save_mode={save_mode!r}.")

        self.max_size: int = max_size
        self.num_fix_frames: int = num_fix_frames
        self.save_mode: str = save_mode
        self.memory: list[MemoryEntry] = []

    # ---------------------------------------------------------------------
    # Container protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.memory)

    def __iter__(self) -> Iterator[MemoryEntry]:
        return iter(self.memory)

    def __getitem__(self, index: int) -> MemoryEntry:
        return self.memory[index]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _trim(self) -> None:
        """Apply FIFO-with-sticky-prefix eviction.

        Keeps the first ``num_fix_frames`` entries unconditionally and retains
        only the most recent entries in the remaining ``max_size -
        num_fix_frames`` budget. Idempotent: calling on a bank that already
        satisfies the size invariant is a no-op.
        """
        if self.max_size <= 0 or len(self.memory) <= self.max_size:
            return
        fixed = self.memory[: self.num_fix_frames]
        tail = self.memory[self.num_fix_frames :]
        keep_tail = max(0, self.max_size - len(fixed))
        # ``tail[-keep_tail:]`` would silently return the whole tail when
        # ``keep_tail == 0`` because Python treats ``a[-0:]`` as ``a[0:]``.
        # Guard explicitly so the sticky prefix can fully saturate the bank.
        if keep_tail == 0:
            self.memory = fixed
        else:
            self.memory = fixed + tail[-keep_tail:]

    # ---------------------------------------------------------------------
    # Mutation API
    # ---------------------------------------------------------------------

    def save_memory_slot(
        self,
        frames: list[PIL.Image.Image],
        audio_latent: Optional[torch.Tensor] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a memory slot, then evict by FIFO-with-sticky-prefix policy.

        Validation:
            * ``frames`` must be a non-empty list of :class:`PIL.Image.Image`.
            * ``audio_latent``, when not ``None``, must be a 3-D tensor of shape
              ``[B, T_a_slot, C]``. It is detached, made contiguous, and moved
              to CPU before storage (matching the source's "store cold, encode
              hot" pattern).

        ``audio_latent=None`` is accepted for forward-compat (e.g. video-only
        memory or an audio-disabled run). Such a slot is silently skipped by
        :meth:`get_memory_audio` and :meth:`get_memory_audio_segment_lengths`
        rather than raising -- a partial audio bank yields no audio memory at
        all, mirroring the source's all-or-nothing behavior at
        ``memory_multishot.py:391-392``.

        Args:
            frames: Per-slot pixel-space clip.
            audio_latent: Optional audio latent ``[B, T_a_slot, C]``.
            metadata: Optional per-slot metadata dict (copied into the entry).
        """
        if not isinstance(frames, list) or len(frames) == 0:
            raise ValueError(
                f"`frames` must be a non-empty list of PIL.Image.Image, got "
                f"type={type(frames).__name__}, "
                f"len={len(frames) if hasattr(frames, '__len__') else 'n/a'}."
            )
        for idx, frame in enumerate(frames):
            if not isinstance(frame, PIL.Image.Image):
                raise TypeError(f"`frames[{idx}]` must be a PIL.Image.Image, got {type(frame).__name__}.")

        prepared_audio: Optional[torch.Tensor] = None
        if audio_latent is not None:
            if not isinstance(audio_latent, torch.Tensor):
                raise TypeError(f"`audio_latent` must be a torch.Tensor or None, got {type(audio_latent).__name__}.")
            if audio_latent.dim() != 3:
                raise ValueError(
                    f"`audio_latent` must have shape [B, T_a_slot, C] (dim=3), got shape={tuple(audio_latent.shape)}."
                )
            prepared_audio = audio_latent.detach().cpu().contiguous()

        entry = MemoryEntry(
            frames=list(frames),
            audio_latent=prepared_audio,
            metadata=dict(metadata) if metadata is not None else {},
        )

        # Append to the free (post-sticky) region, then trim. This matches the
        # source order (memory_multishot.py:376-380) and ensures the sticky
        # prefix is never re-shuffled by a push.
        fixed = self.memory[: self.num_fix_frames]
        free = self.memory[self.num_fix_frames :]
        free.append(entry)
        self.memory = fixed + free
        self._trim()

    def clear(self) -> None:
        """Reset the bank to an empty state. Useful between samples and in tests."""
        self.memory = []

    # ---------------------------------------------------------------------
    # Read API
    # ---------------------------------------------------------------------

    def get_memory_frames(self) -> list[list[PIL.Image.Image]]:
        """Return per-slot pixel-frame lists in slot order.

        The outer list has length ``len(self)``; each inner list is the slot's
        clip (typically ``video_clip_num_frames`` frames in the source recipe).
        Cross-slot flattening is left to the pipeline because the wrapper needs
        slot boundaries for downstream audio-aware clip selection.
        """
        return [entry.frames for entry in self.memory]

    def get_memory_audio(self) -> Optional[torch.Tensor]:
        """Concatenate per-slot audio latents along the time axis.

        Returns ``None`` when the bank is empty OR when ANY slot has
        ``audio_latent=None`` (matching the source's all-or-nothing policy at
        ``memory_multishot.py:391-392``). All non-``None`` slots must agree on
        batch and channel dimensions; mismatches raise ``ValueError``.

        Returns:
            Tensor of shape ``[B, sum(T_a_slot), C]`` on CPU, or ``None``.
        """
        if len(self.memory) == 0:
            return None
        audio_latents = [entry.audio_latent for entry in self.memory]
        if any(audio_latent is None for audio_latent in audio_latents):
            return None

        first = audio_latents[0]
        assert first is not None  # for type-checkers; guarded by the any() above
        batch_size = first.shape[0]
        channels = first.shape[2]
        for audio_latent in audio_latents:
            assert audio_latent is not None
            if audio_latent.shape[0] != batch_size or audio_latent.shape[2] != channels:
                raise ValueError(
                    "All memory audio latents must share batch and channel dimensions, "
                    f"got first={tuple(first.shape)}, current={tuple(audio_latent.shape)}."
                )
        return torch.cat(audio_latents, dim=1).contiguous()

    def get_memory_audio_segment_lengths(self) -> tuple[tuple[int, ...], ...]:
        """Return per-slot audio-latent lengths in source-compatible nesting.

        Outer tuple length is ``1`` by convention (per the audio spec D.0 Sec.
        1, and matching ``memory_multishot.py:406-410``); the wrapper indexes
        ``[batch_idx=0]`` to retrieve the inner tuple of per-slot ``T_a``
        values. Empty when the bank is empty OR any slot lacks an audio
        latent (consistent with :meth:`get_memory_audio` returning ``None`` in
        the same situation).
        """
        if len(self.memory) == 0:
            return ()
        audio_latents = [entry.audio_latent for entry in self.memory]
        if any(audio_latent is None for audio_latent in audio_latents):
            return ()
        return (tuple(int(audio_latent.shape[1]) for audio_latent in audio_latents if audio_latent is not None),)
