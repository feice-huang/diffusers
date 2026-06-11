# Copyright 2026 The Lightricks team, The JD-AI team, and The HuggingFace Team.
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
"""End-to-end multi-shot inference demo for ``JoyAIEchoPipeline``.

Loops over a list of per-shot prompts, runs the DMD 8-step CFG-free schedule on
each, threads a shared ``JoyAIEchoMemoryBank`` across calls so the bank
accumulates paired audio-video memory, writes per-shot mp4s, and concatenates
them into a single output.

REQUIREMENTS:
  * The diffusers checkpoint directory must already exist (run
    ``scripts/convert_joyai_echo_to_diffusers.py`` first).
  * ``ffmpeg`` available on PATH (used by ``concat_video_files``).

Example::

    python scripts/run_joyai_echo_e2e.py \\
        --ckpt /path/to/JoyAI-Echo-diffusers \\
        --prompts /path/to/prompts.json \\
        --out /tmp/joyai_echo_e2e
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diffusers import JoyAIEchoMemoryBank, JoyAIEchoPipeline
from diffusers.utils import concat_video_files, encode_video


# Reproduces JoyAI-Echo's ``configs/inference.yaml`` defaults.
NUM_FRAMES = 241
HEIGHT = 736
WIDTH = 1280
FPS = 25
SEED = 12345
DTYPE = torch.bfloat16
DEVICE = "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True, help="Diffusers JoyAI-Echo checkpoint directory.")
    parser.add_argument("--prompts", type=Path, required=True, help="Prompts JSON file with a top-level `prompts` list.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-shot + combined mp4s.")
    parser.add_argument("--seed", type=int, default=SEED, help="Base seed; shot N uses `seed + N`.")
    args = parser.parse_args()

    if not (args.ckpt / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Missing {args.ckpt / 'model_index.json'}. "
            "Run scripts/convert_joyai_echo_to_diffusers.py first."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = json.loads(args.prompts.read_text())["prompts"]
    print(f"[e2e] loaded {len(prompts)} prompts")

    pipe = JoyAIEchoPipeline.from_pretrained(str(args.ckpt), torch_dtype=DTYPE).to(DEVICE)
    audio_sample_rate = pipe.vocoder.config.output_sampling_rate
    sigmas = list(JoyAIEchoPipeline.DMD_SIGMAS)[:-1]  # 8 transitions; trailing 0 auto-appended

    memory_bank = JoyAIEchoMemoryBank()
    shot_paths: list[Path] = []
    t0 = time.perf_counter()

    for shot_idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed + shot_idx)
        print(f"[e2e] shot {shot_idx + 1}/{len(prompts)} mem={len(memory_bank)} '{prompt[:60]}...'")
        ts = time.perf_counter()

        video, audio = pipe(
            prompt=prompt,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=float(FPS),
            num_inference_steps=None,
            sigmas=sigmas,
            guidance_scale=1.0,  # DMD is CFG-free
            audio_guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
            memory_bank=memory_bank,
        )

        shot_path = args.out / f"shot_{shot_idx:03d}.mp4"
        encode_video(
            video[0],
            fps=FPS,
            audio=audio[0].float().cpu(),
            audio_sample_rate=audio_sample_rate,
            output_path=str(shot_path),
        )
        shot_paths.append(shot_path)
        print(f"[e2e] shot {shot_idx + 1} done in {time.perf_counter() - ts:.1f}s")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    combined = args.out / f"combined_shots_seed{args.seed}.mp4"
    concat_video_files([str(p) for p in shot_paths], str(combined))
    print(f"[e2e] DONE in {time.perf_counter() - t0:.1f}s — {combined}")


if __name__ == "__main__":
    main()
