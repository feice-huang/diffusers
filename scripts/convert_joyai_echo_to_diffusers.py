# Copyright 2025 The Lightricks team, The JD-AI team, and The HuggingFace Team.
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
"""Convert a JoyAI-Echo monolithic safetensors release into a diffusers-style component layout.

JoyAI-Echo ships its full multimodal video+audio diffusion stack as a single ~46 GB
``JoyAI-Echo-release.safetensors`` (5947 keys, 5 top-level prefixes:
``model.diffusion_model.*``, ``vae.*``, ``audio_vae.*``, ``vocoder.*``,
``text_embedding_projection.*``). Architecturally this is an LTX-2.3 variant — every
weight maps 1:1 onto an existing diffusers class once the right rename rules are applied:

  * ``model.diffusion_model.*`` (sans connector keys) -> ``JoyAIEchoTransformer3DModel``
    (mirrors ``LTX2VideoTransformer3DModel``)
  * ``model.diffusion_model.{video,audio}_embeddings_connector.*`` +
    ``text_embedding_projection.*`` -> ``LTX2TextConnectors``
  * ``vae.*`` -> ``AutoencoderKLLTX2Video``
  * ``audio_vae.*`` -> ``AutoencoderKLLTX2Audio``
  * ``vocoder.*`` -> ``LTX2VocoderWithBWE``

All rename rules and target ``__init__`` kwargs reuse the LTX-2.3 versions of
``convert_ltx2_to_diffusers.py``; this script wraps them with JoyAI-Echo-specific
defaults plus Gemma text-encoder symlinking and a ``model_index.json`` for the
``JoyAIEchoPipeline``.

Example::

    python scripts/convert_joyai_echo_to_diffusers.py \\
        --src /path/to/jdopensource/JoyAI-Echo \\
        --gemma-src /path/to/google/gemma-3-12b-it \\
        --out /path/to/JoyAI-Echo-diffusers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import safetensors
import torch
from accelerate import init_empty_weights


# Reuse the LTX-2.3 rename dicts, configs, and helpers verbatim. Per project
# memory ("base LTX2 untouched") we MUST NOT modify any of these — only import.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from convert_ltx2_to_diffusers import (  # noqa: E402
    LTX_2_0_AUDIO_VAE_RENAME_DICT,
    LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP,
    LTX_2_0_CONNECTORS_SPECIAL_KEYS_REMAP,
    LTX_2_0_VAE_SPECIAL_KEYS_REMAP,
    LTX_2_3_CONNECTORS_KEYS_RENAME_DICT,
    LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT,
    LTX_2_3_VIDEO_VAE_RENAME_DICT,
    LTX_2_3_VOCODER_RENAME_DICT,
    LTX_2_3_VOCODER_SPECIAL_KEYS_REMAP,
    convert_ltx2_transformer_adaln_single,
    get_ltx2_audio_vae_config,
    get_ltx2_connectors_config,
    get_ltx2_transformer_config,
    get_ltx2_video_vae_config,
    get_ltx2_vocoder_config,
)

from diffusers import (  # noqa: E402
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors, LTX2VocoderWithBWE  # noqa: E402
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402


# Optional import of the JoyAI-Echo transformer class. The class file at
# ``diffusers/models/transformers/transformer_ltx2_joyai_echo.py`` is being written
# in parallel (TX.1) and may be temporarily un-importable. The state_dict shape is
# identical to ``LTX2VideoTransformer3DModel`` so the base class is a safe proxy
# for building rename dicts and writing ``transformer/config.json``.
try:
    from diffusers.models.transformers.transformer_ltx2_joyai_echo import (
        JoyAIEchoTransformer3DModel,
    )

    TRANSFORMER_CLASS = JoyAIEchoTransformer3DModel
    TRANSFORMER_CLASS_NAME = "JoyAIEchoTransformer3DModel"
    _JOYAI_TRANSFORMER_IMPORTED = True
except Exception as exc:  # noqa: BLE001 — class file may be syntactically broken
    warnings.warn(
        f"Could not import JoyAIEchoTransformer3DModel ({exc!r}); falling back to "
        "LTX2VideoTransformer3DModel as a structural proxy. The state_dict layouts "
        "are identical, so rename/config logic is unaffected, but the saved "
        "transformer/config.json will still record the JoyAI class name so that the "
        "pipeline loads the correct class once TX.1 lands.",
        RuntimeWarning,
        stacklevel=1,
    )
    TRANSFORMER_CLASS = LTX2VideoTransformer3DModel
    TRANSFORMER_CLASS_NAME = "JoyAIEchoTransformer3DModel"
    _JOYAI_TRANSFORMER_IMPORTED = False


# ----------------------------------------------------------------------------
# Constants — pulled from A.1 keymap report (cross-checked by A.2 review).
# ----------------------------------------------------------------------------

SOURCE_SAFETENSORS_BASENAME = "JoyAI-Echo-release.safetensors"

TOP_LEVEL_PREFIXES = ("model", "vae", "audio_vae", "vocoder", "text_embedding_projection")
EXPECTED_SOURCE_KEY_COUNT = 5947  # for the post-conversion summary sanity check

# ``LTX2_3``-style configs from ``convert_ltx2_to_diffusers.py`` already match
# JoyAI-Echo's tensor shapes byte-for-byte (verified during A.3 dev). We pull
# them via ``get_ltx2_*_config("2.3", ...)`` rather than redeclaring inline so
# any future LTX-2.3 config fix automatically propagates here.


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _update_state_dict_inplace(state_dict: dict[str, Any], old_key: str, new_key: str) -> None:
    if old_key == new_key:
        return
    state_dict[new_key] = state_dict.pop(old_key)


def _strip_prefix_once(key: str, prefix: str) -> str:
    """Strip ``prefix`` from ``key`` exactly once (only if at the start).

    This is the safe alternative to ``str.replace`` for outer-namespace strips
    where the inner namespace may contain the same string (e.g. the vocoder
    ``vocoder.vocoder.*`` nesting flagged by A.2 NIT #2).
    """
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def _apply_rename_dict(state_dict: dict[str, Any], rename_dict: dict[str, str]) -> None:
    """Apply each rename rule globally with ``str.replace``, in dict order.

    This mirrors the LTX-2 conversion idiom in ``convert_ltx2_to_diffusers.py``
    and is correct when rename keys do not overlap each other in ways the
    iteration order can corrupt. The JoyAI-Echo conversion guarantees this
    invariant because outer-namespace prefixes are stripped via
    ``_strip_prefix_once`` BEFORE this function runs.
    """
    for old_key in list(state_dict.keys()):
        new_key = old_key
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        _update_state_dict_inplace(state_dict, old_key, new_key)


def _apply_special_keys(state_dict: dict[str, Any], special_keys_remap: dict[str, Any]) -> None:
    for key in list(state_dict.keys()):
        for special_key, handler_fn_inplace in special_keys_remap.items():
            if special_key not in key:
                continue
            handler_fn_inplace(key, state_dict)


# ----------------------------------------------------------------------------
# Per-bucket conversion functions
# ----------------------------------------------------------------------------


def convert_transformer(src_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert ``model.diffusion_model.*`` (minus connector keys) -> transformer state_dict.

    Routing rule: the input ``src_state_dict`` may contain
    ``video_embeddings_connector`` / ``audio_embeddings_connector`` keys; per A.2's
    "single_file_utils drop hazard" finding we must NOT route those through the
    transformer (the base LTX-2 special-keys table would silently drop them).
    They are skipped here with a ``startswith`` filter — connector keys are
    handled by :func:`convert_connectors` instead. Likewise,
    ``text_embedding_projection.*`` keys (which co-live at the top level of the
    source safetensors with ``model.diffusion_model.*``) are also routed only
    via the connector path and excluded here.

    Other rules (all already in ``LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT``):
      * ``model.diffusion_model.`` -> ``""`` (prefix strip — done before rename)
      * ``patchify_proj`` -> ``proj_in``, ``audio_patchify_proj`` -> ``audio_proj_in``
      * ``q_norm`` -> ``norm_q``, ``k_norm`` -> ``norm_k``
      * ``prompt_adaln_single`` -> ``prompt_adaln``, ``audio_prompt_adaln_single`` -> ``audio_prompt_adaln``
      * ``av_ca_*_adaln_single`` -> ``av_cross_attn_*``
      * ``scale_shift_table_a2v_ca_video|audio`` -> ``video|audio_a2v_cross_attn_scale_shift_table``

    Per A.2 NIT #4: ``prompt_adaln_single`` is a substring of
    ``audio_prompt_adaln_single``. The base ``LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT``
    lists ``audio_prompt_adaln_single`` BEFORE ``prompt_adaln_single`` so the
    longer key renames first — verified upstream.

    Per A.2 NIT #3: ``use_prompt_embeddings=False`` is in the LTX-2.3 transformer
    config dict (``convert_ltx2_to_diffusers.py:364``), so no extra handling needed.
    """
    converted: dict[str, Any] = {}
    connector_inner_prefixes = ("video_embeddings_connector.", "audio_embeddings_connector.")
    for key, value in src_state_dict.items():
        # Only ``model.diffusion_model.*`` (excluding connector keys) is routed
        # through the transformer bucket. ``text_embedding_projection.*`` and
        # the two ``*_embeddings_connector.*`` namespaces belong to the
        # connector bucket (see :func:`convert_connectors`).
        if not key.startswith("model.diffusion_model."):
            continue
        stripped = key[len("model.diffusion_model.") :]
        if stripped.startswith(connector_inner_prefixes):
            continue
        converted[stripped] = value

    _apply_rename_dict(converted, LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT)
    # The adaln_single helper is the only piece of the base
    # LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP that we want — the other two entries
    # (video_embeddings_connector, audio_embeddings_connector via remove_keys_inplace)
    # are intentionally NOT applied because we've already routed those keys
    # to the connector bucket.
    _apply_special_keys(converted, {"adaln_single": convert_ltx2_transformer_adaln_single})
    return converted


def convert_connectors(src_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert connector + text-embedding-projection sources -> connectors state_dict.

    Input sources:
      * ``model.diffusion_model.{video,audio}_embeddings_connector.*`` (258 keys)
      * ``text_embedding_projection.{video,audio}_aggregate_embed.*`` (4 keys)

    Rules (all already in ``LTX_2_3_CONNECTORS_KEYS_RENAME_DICT``):
      * ``connectors.`` -> ``""`` (no-op for JoyAI sources, kept for parity)
      * ``video_embeddings_connector`` -> ``video_connector``
      * ``audio_embeddings_connector`` -> ``audio_connector``
      * ``transformer_1d_blocks`` -> ``transformer_blocks``
      * ``text_embedding_projection.{video,audio}_aggregate_embed`` -> ``{video,audio}_text_proj_in``
      * ``q_norm`` -> ``norm_q``, ``k_norm`` -> ``norm_k``
    """
    converted: dict[str, Any] = {}
    for key, value in src_state_dict.items():
        if key.startswith("text_embedding_projection."):
            # Top-level — keep as-is for the global rename dict to match.
            converted[key] = value
        elif key.startswith("model.diffusion_model.video_embeddings_connector.") or key.startswith(
            "model.diffusion_model.audio_embeddings_connector."
        ):
            # Strip the ``model.diffusion_model.`` outer prefix; the inner
            # ``{video,audio}_embeddings_connector`` substring is then renamed
            # by ``LTX_2_3_CONNECTORS_KEYS_RENAME_DICT`` to ``{video,audio}_connector``.
            converted[key[len("model.diffusion_model.") :]] = value
        # Silently skip everything else — transformer keys belong to
        # :func:`convert_transformer`, not here.

    _apply_rename_dict(converted, LTX_2_3_CONNECTORS_KEYS_RENAME_DICT)
    _apply_special_keys(converted, LTX_2_0_CONNECTORS_SPECIAL_KEYS_REMAP)
    return converted


def convert_vae(src_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert ``vae.*`` -> AutoencoderKLLTX2Video state_dict.

    Reuses LTX-2.3 rename dict (which extends LTX-2.0 with the extra
    ``up_blocks.7/8`` decoder groups present in JoyAI-Echo's source).

    Per A.2 NIT #1: the LTX-2.3 ``get_ltx2_video_vae_config`` already encodes
    the correct ``layers_per_block=(4, 6, 4, 2, 2)`` /
    ``decoder_layers_per_block=(4, 6, 4, 2, 2)`` cascades for JoyAI-Echo, so we
    do not declare a JoyAI-specific config inline — we just delegate.
    Verified during A.3 dev: 170 source keys -> 170 target keys, 0 diff.
    """
    converted = {_strip_prefix_once(k, "vae."): v for k, v in src_state_dict.items()}
    _apply_rename_dict(converted, LTX_2_3_VIDEO_VAE_RENAME_DICT)
    _apply_special_keys(converted, LTX_2_0_VAE_SPECIAL_KEYS_REMAP)
    return converted


def convert_audio_vae(src_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert ``audio_vae.*`` -> AutoencoderKLLTX2Audio state_dict.

    Per A.2 PASS: 102 source keys -> 102 target keys with the existing
    ``LTX_2_0_AUDIO_VAE_RENAME_DICT``.
    """
    converted = {_strip_prefix_once(k, "audio_vae."): v for k, v in src_state_dict.items()}
    _apply_rename_dict(converted, LTX_2_0_AUDIO_VAE_RENAME_DICT)
    _apply_special_keys(converted, LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP)
    return converted


def convert_vocoder(src_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert ``vocoder.*`` -> LTX2VocoderWithBWE state_dict.

    Per A.2 NIT #2: the source has a nested ``vocoder.vocoder.*`` namespace.
    A naive ``str.replace("vocoder.", "")`` would strip BOTH the outer prefix
    AND the inner one — destroying the inner ``vocoder.*`` keys (the target's
    ``LTX2VocoderWithBWE.vocoder.*`` submodule). Fix: strip the outer prefix
    EXACTLY ONCE with ``_strip_prefix_once``, THEN apply the global renames.

    Empirical regression test in ``_run_smoke_tests`` asserts that
    ``vocoder.vocoder.foo`` becomes ``vocoder.foo`` (NOT ``foo``).
    """
    converted = {_strip_prefix_once(k, "vocoder."): v for k, v in src_state_dict.items()}
    _apply_rename_dict(converted, LTX_2_3_VOCODER_RENAME_DICT)
    _apply_special_keys(converted, LTX_2_3_VOCODER_SPECIAL_KEYS_REMAP)
    return converted


# ----------------------------------------------------------------------------
# Component instantiation + save
# ----------------------------------------------------------------------------


def _instantiate_and_load(cls, config: dict[str, Any], state_dict: dict[str, Any], name: str):
    """Instantiate ``cls`` with ``config``, load ``state_dict`` strict=True, and return it.

    Errors are augmented with the bucket name so the orchestrator (A.4) can
    quickly identify which component failed.
    """
    with init_empty_weights():
        model = cls.from_config(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"[{name}] state_dict mismatch — missing={list(missing)[:10]} "
            f"(total {len(missing)}), unexpected={list(unexpected)[:10]} "
            f"(total {len(unexpected)})"
        )
    return model


def _link_or_copy_gemma(gemma_src: Path, out_dir: Path, copy: bool) -> None:
    """Materialize Gemma artifacts into the diffusers ``text_encoder/``, ``tokenizer/``,
    ``processor/`` subdirs.

    For now we mirror the Gemma source folder into each of the three subdirs,
    which is wasteful but matches diffusers' loader expectations (each component
    is loaded with ``from_pretrained(<subdir>)``). The expectation is that A.4
    will refine which Gemma files are needed per role; this step just guarantees
    a working layout for end-to-end testing.
    """
    if not gemma_src.exists():
        raise FileNotFoundError(f"--gemma-src does not exist: {gemma_src}")

    for subdir in ("text_encoder", "tokenizer", "processor"):
        dest = out_dir / subdir
        dest.mkdir(parents=True, exist_ok=True)
        for entry in gemma_src.iterdir():
            target = dest / entry.name
            if target.exists() or target.is_symlink():
                target.unlink()
            if copy:
                if entry.is_dir():
                    shutil.copytree(entry, target)
                else:
                    shutil.copy2(entry, target)
            else:
                target.symlink_to(entry.resolve())


def _write_model_index(out_dir: Path) -> None:
    """Write the diffusers ``model_index.json`` for JoyAIEchoPipeline.

    Components listed match the JoyAIEchoPipeline ``register_modules`` call
    (``pipeline_joyai_echo.py:209-219``): scheduler, vae, audio_vae, text_encoder,
    tokenizer, connectors, transformer, vocoder, processor.

    The processor is marked optional in the pipeline class but we still emit it
    here for completeness; downstream code can drop the directory if a Gemma3
    processor isn't required.
    """
    model_index = {
        "_class_name": "JoyAIEchoPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "vae": ["diffusers", "AutoencoderKLLTX2Video"],
        "audio_vae": ["diffusers", "AutoencoderKLLTX2Audio"],
        "text_encoder": ["transformers", "Gemma3ForConditionalGeneration"],
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "connectors": ["diffusers", "LTX2TextConnectors"],
        "transformer": ["diffusers", TRANSFORMER_CLASS_NAME],
        "vocoder": ["diffusers", "LTX2VocoderWithBWE"],
        "processor": ["transformers", "Gemma3Processor"],
    }
    with open(out_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)


# ----------------------------------------------------------------------------
# Smoke tests (run with --smoke-test; do NOT touch the 46GB checkpoint)
# ----------------------------------------------------------------------------


def _make_dummy_tensor() -> torch.Tensor:
    return torch.zeros(1)


def _run_smoke_tests() -> int:
    """Run the six verification steps the task brief requires.

    Returns a process exit code (0 on full PASS, 1 on any FAIL).
    """
    failures: list[str] = []

    # === D. Vocoder regression test ===
    voc_sd = {
        "vocoder.vocoder.foo": _make_dummy_tensor(),
        "vocoder.bwe_generator.bar": _make_dummy_tensor(),
        "vocoder.mel_stft.baz": _make_dummy_tensor(),
    }
    out = convert_vocoder(voc_sd)
    expected = {"vocoder.foo", "bwe_generator.bar", "mel_stft.baz"}
    if set(out.keys()) != expected:
        failures.append(f"[D vocoder] expected {expected}, got {set(out.keys())}")
        print(f"  [FAIL] vocoder regression: got {set(out.keys())}")
    else:
        print("  [PASS] vocoder regression — vocoder.vocoder.foo -> vocoder.foo (inner namespace preserved)")

    # === E. prompt_adaln_single regression test ===
    xfmr_sd = {
        "model.diffusion_model.prompt_adaln_single.X": _make_dummy_tensor(),
        "model.diffusion_model.audio_prompt_adaln_single.X": _make_dummy_tensor(),
    }
    out = convert_transformer(xfmr_sd)
    expected = {"prompt_adaln.X", "audio_prompt_adaln.X"}
    if set(out.keys()) != expected:
        failures.append(f"[E prompt_adaln] expected {expected}, got {set(out.keys())}")
        print(f"  [FAIL] prompt_adaln regression: got {set(out.keys())}")
    else:
        print("  [PASS] prompt_adaln regression — no substring collision")

    # === F. Connector regression test ===
    conn_sd = {
        "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": _make_dummy_tensor(),
        "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": _make_dummy_tensor(),
        "model.diffusion_model.video_embeddings_connector.learnable_registers": _make_dummy_tensor(),
        "model.diffusion_model.audio_embeddings_connector.learnable_registers": _make_dummy_tensor(),
        "text_embedding_projection.video_aggregate_embed.weight": _make_dummy_tensor(),
        "text_embedding_projection.audio_aggregate_embed.weight": _make_dummy_tensor(),
    }
    conn_out = convert_connectors(conn_sd)
    conn_expected = {
        "video_connector.transformer_blocks.0.attn1.to_q.weight",
        "audio_connector.transformer_blocks.0.attn1.to_q.weight",
        "video_connector.learnable_registers",
        "audio_connector.learnable_registers",
        "video_text_proj_in.weight",
        "audio_text_proj_in.weight",
    }
    if set(conn_out.keys()) != conn_expected:
        failures.append(f"[F connectors] expected {conn_expected}, got {set(conn_out.keys())}")
        print(f"  [FAIL] connector regression: got {set(conn_out.keys())}")
    else:
        print("  [PASS] connector regression — connector keys route to connectors bucket")

    # Bonus: assert convert_transformer DOES NOT emit any connector keys when
    # the same connector sources are passed through it.
    xfmr_out = convert_transformer(conn_sd)
    if any("connector" in k for k in xfmr_out.keys()):
        leaked = [k for k in xfmr_out.keys() if "connector" in k]
        failures.append(f"[F connectors] leaked into transformer: {leaked}")
        print(f"  [FAIL] connector keys leaked into transformer bucket: {leaked}")
    else:
        print("  [PASS] convert_transformer correctly excludes connector keys")

    # === C. Smoke test with mock keys for each convert_*: just verify the
    # routing functions emit the right *set* of keys vs target class state_dict ===
    # NOTE: this is a structural ID check; not a full conversion of the 46GB blob.
    #
    # We synthesize a representative subset of source keys for each bucket and
    # confirm that the converted keys form a SUBSET of the target class's
    # state_dict. (Full equality is verified separately in A.4 by loading the
    # real checkpoint.)

    def _check_subset(name: str, converted_keys: set[str], target_keys: set[str]) -> None:
        leaks = converted_keys - target_keys
        if leaks:
            failures.append(f"[C {name}] {len(leaks)} converted keys not in target: {sorted(leaks)[:5]}")
            print(f"  [FAIL] {name} converted keys not in target: {sorted(leaks)[:5]}")
        else:
            print(f"  [PASS] {name} mock conversion ({len(converted_keys)} keys) all in target ({len(target_keys)})")

    # Transformer subset
    transformer_config, _, _ = get_ltx2_transformer_config("2.3")
    with init_empty_weights():
        transformer_model = LTX2VideoTransformer3DModel.from_config(transformer_config["diffusers_config"])
    transformer_target_keys = set(transformer_model.state_dict().keys())
    # Mock keys are drawn from real source patterns (verified during A.3 dev
    # against the actual 5947-key safetensors). In particular,
    # ``scale_shift_table_a2v_ca_{video,audio}`` is a per-BLOCK key in the
    # source (lives under ``transformer_blocks.{i}.``), not a top-level key.
    transformer_mock = {
        "model.diffusion_model.adaln_single.linear.weight": _make_dummy_tensor(),
        "model.diffusion_model.audio_adaln_single.linear.weight": _make_dummy_tensor(),
        "model.diffusion_model.prompt_adaln_single.linear.weight": _make_dummy_tensor(),
        "model.diffusion_model.audio_prompt_adaln_single.linear.weight": _make_dummy_tensor(),
        "model.diffusion_model.patchify_proj.weight": _make_dummy_tensor(),
        "model.diffusion_model.audio_patchify_proj.weight": _make_dummy_tensor(),
        "model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight": _make_dummy_tensor(),
        "model.diffusion_model.transformer_blocks.0.scale_shift_table_a2v_ca_video": _make_dummy_tensor(),
        "model.diffusion_model.av_ca_video_scale_shift_adaln_single.linear.weight": _make_dummy_tensor(),
    }
    _check_subset("transformer", set(convert_transformer(transformer_mock).keys()), transformer_target_keys)

    # Connectors subset
    connectors_config, _, _ = get_ltx2_connectors_config("2.3")
    with init_empty_weights():
        connectors_model = LTX2TextConnectors.from_config(connectors_config["diffusers_config"])
    connectors_target_keys = set(connectors_model.state_dict().keys())
    _check_subset("connectors", set(conn_out.keys()), connectors_target_keys)

    # VAE subset
    vae_config, _, _ = get_ltx2_video_vae_config("2.3", timestep_conditioning=False)
    with init_empty_weights():
        vae_model = AutoencoderKLLTX2Video.from_config(vae_config["diffusers_config"])
    vae_target_keys = set(vae_model.state_dict().keys())
    vae_mock = {
        "vae.encoder.conv_in.conv.weight": _make_dummy_tensor(),
        "vae.encoder.down_blocks.0.res_blocks.0.conv1.conv.weight": _make_dummy_tensor(),
        "vae.encoder.down_blocks.8.res_blocks.0.conv1.conv.weight": _make_dummy_tensor(),
        "vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight": _make_dummy_tensor(),
        "vae.decoder.up_blocks.8.res_blocks.0.conv1.conv.weight": _make_dummy_tensor(),
        "vae.per_channel_statistics.mean-of-means": _make_dummy_tensor(),
        "vae.per_channel_statistics.std-of-means": _make_dummy_tensor(),
    }
    _check_subset("vae", set(convert_vae(vae_mock).keys()), vae_target_keys)

    # Audio VAE subset
    audio_vae_config, _, _ = get_ltx2_audio_vae_config("2.3")
    with init_empty_weights():
        audio_vae_model = AutoencoderKLLTX2Audio.from_config(audio_vae_config["diffusers_config"])
    audio_vae_target_keys = set(audio_vae_model.state_dict().keys())
    audio_vae_mock = {
        "audio_vae.per_channel_statistics.mean-of-means": _make_dummy_tensor(),
        "audio_vae.per_channel_statistics.std-of-means": _make_dummy_tensor(),
    }
    _check_subset("audio_vae", set(convert_audio_vae(audio_vae_mock).keys()), audio_vae_target_keys)

    # Vocoder subset (uses the regression input above)
    vocoder_config, _, _ = get_ltx2_vocoder_config("2.3")
    with init_empty_weights():
        vocoder_model = LTX2VocoderWithBWE.from_config(vocoder_config["diffusers_config"])
    vocoder_target_keys = set(vocoder_model.state_dict().keys())
    # ``mel_stft`` keys live under ``vocoder.mel_stft.{mel_basis,stft_fn.forward_basis,
    # stft_fn.inverse_basis}`` in the source (3 keys total). The target
    # LTX2VocoderWithBWE exposes these under the same suffix paths.
    vocoder_mock = {
        "vocoder.vocoder.conv_pre.weight": _make_dummy_tensor(),
        "vocoder.bwe_generator.conv_pre.weight": _make_dummy_tensor(),
        "vocoder.mel_stft.mel_basis": _make_dummy_tensor(),
    }
    _check_subset("vocoder", set(convert_vocoder(vocoder_mock).keys()), vocoder_target_keys)

    if failures:
        print()
        print("SMOKE TEST FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print()
    print("All smoke tests PASS.")
    return 0


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------


def _load_source(src_path: Path, dtype: torch.dtype | None) -> dict[str, dict[str, torch.Tensor]]:
    """Stream-load the source safetensors into 5 per-prefix dicts.

    Each tensor is read once via ``safe_open.get_tensor`` and bucketed by its
    top-level prefix. This avoids holding the full 46 GB in a single flat
    state_dict, although peak memory is still O(46 GB) since we keep every
    tensor alive until each component is saved. Future optimization: write
    components out one-at-a-time and drop refs after each save.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source safetensors not found: {src_path}")

    buckets: dict[str, dict[str, torch.Tensor]] = {p: {} for p in TOP_LEVEL_PREFIXES}
    with safetensors.safe_open(str(src_path), framework="pt") as f:
        for key in f.keys():
            top = key.split(".", 1)[0]
            if top not in buckets:
                raise ValueError(f"Unknown top-level prefix for key {key!r}: {top!r}")
            tensor = f.get_tensor(key)
            if dtype is not None:
                tensor = tensor.to(dtype)
            buckets[top][key] = tensor
    return buckets


def _resolve_dtype(arg: str | None) -> torch.dtype | None:
    if arg is None or arg.lower() == "source":
        return None
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return mapping[arg.lower()]


def main(args: argparse.Namespace) -> int:
    if args.smoke_test:
        return _run_smoke_tests()

    src_dir = Path(args.src)
    if src_dir.is_file():
        src_path = src_dir
    else:
        src_path = src_dir / SOURCE_SAFETENSORS_BASENAME
    out_dir = Path(args.out)
    gemma_src = Path(args.gemma_src)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(args.dtype)
    print(f"[A.3 convert] src={src_path}  out={out_dir}  dtype={args.dtype}")
    print("[A.3 convert] streaming safetensors (this allocates ~46 GB) ...")
    buckets = _load_source(src_path, dtype)

    n_total = sum(len(v) for v in buckets.values())
    print(
        f"[A.3 convert] loaded {n_total} keys; bucket counts: "
        + ", ".join(f"{k}={len(v)}" for k, v in buckets.items())
    )
    if n_total != EXPECTED_SOURCE_KEY_COUNT:
        warnings.warn(
            f"Source has {n_total} keys, expected {EXPECTED_SOURCE_KEY_COUNT}. Proceeding anyway.",
            RuntimeWarning,
        )

    # Merge model.* and text_embedding_projection.* for the dit/connectors stage.
    dit_and_text_proj = {**buckets["model"], **buckets["text_embedding_projection"]}

    # -- transformer --------------------------------------------------------
    transformer_config, _, _ = get_ltx2_transformer_config("2.3")
    transformer_sd = convert_transformer(dit_and_text_proj)
    print(f"[A.3 convert] transformer: {len(transformer_sd)} keys")
    transformer = _instantiate_and_load(
        TRANSFORMER_CLASS, transformer_config["diffusers_config"], transformer_sd, "transformer"
    )
    transformer.save_pretrained(out_dir / "transformer", safe_serialization=True, max_shard_size=args.shard_size)

    # -- connectors ---------------------------------------------------------
    connectors_config, _, _ = get_ltx2_connectors_config("2.3")
    connectors_sd = convert_connectors(dit_and_text_proj)
    print(f"[A.3 convert] connectors: {len(connectors_sd)} keys")
    connectors = _instantiate_and_load(
        LTX2TextConnectors, connectors_config["diffusers_config"], connectors_sd, "connectors"
    )
    connectors.save_pretrained(out_dir / "connectors", safe_serialization=True, max_shard_size=args.shard_size)

    # -- video VAE ----------------------------------------------------------
    vae_config, _, _ = get_ltx2_video_vae_config("2.3", timestep_conditioning=False)
    vae_sd = convert_vae(buckets["vae"])
    print(f"[A.3 convert] vae: {len(vae_sd)} keys")
    vae = _instantiate_and_load(AutoencoderKLLTX2Video, vae_config["diffusers_config"], vae_sd, "vae")
    vae.save_pretrained(out_dir / "vae", safe_serialization=True, max_shard_size=args.shard_size)

    # -- audio VAE ----------------------------------------------------------
    audio_vae_config, _, _ = get_ltx2_audio_vae_config("2.3")
    audio_vae_sd = convert_audio_vae(buckets["audio_vae"])
    print(f"[A.3 convert] audio_vae: {len(audio_vae_sd)} keys")
    audio_vae = _instantiate_and_load(
        AutoencoderKLLTX2Audio, audio_vae_config["diffusers_config"], audio_vae_sd, "audio_vae"
    )
    audio_vae.save_pretrained(out_dir / "audio_vae", safe_serialization=True, max_shard_size=args.shard_size)

    # -- vocoder ------------------------------------------------------------
    vocoder_config, _, _ = get_ltx2_vocoder_config("2.3")
    vocoder_sd = convert_vocoder(buckets["vocoder"])
    print(f"[A.3 convert] vocoder: {len(vocoder_sd)} keys")
    vocoder = _instantiate_and_load(LTX2VocoderWithBWE, vocoder_config["diffusers_config"], vocoder_sd, "vocoder")
    vocoder.save_pretrained(out_dir / "vocoder", safe_serialization=True, max_shard_size=args.shard_size)

    # -- Gemma text encoder / tokenizer / processor -------------------------
    print(f"[A.3 convert] linking gemma artifacts from {gemma_src}")
    _link_or_copy_gemma(gemma_src, out_dir, copy=args.copy_gemma)

    # -- pipeline model_index.json ------------------------------------------
    _write_model_index(out_dir)
    FlowMatchEulerDiscreteScheduler().save_pretrained(out_dir / "scheduler")

    # -- summary ------------------------------------------------------------
    total = len(transformer_sd) + len(connectors_sd) + len(vae_sd) + len(audio_vae_sd) + len(vocoder_sd)
    print()
    print("=" * 72)
    print(
        f"[A.3 convert] wrote {len(transformer_sd)} transformer + "
        f"{len(connectors_sd)} connector + {len(vae_sd)} vae + "
        f"{len(audio_vae_sd)} audio_vae + {len(vocoder_sd)} vocoder = {total} keys"
    )
    print(f"[A.3 convert] source had {n_total} keys (expected {EXPECTED_SOURCE_KEY_COUNT})")
    if total != n_total:
        print(f"[A.3 convert] WARNING: source-total {n_total} != routed-total {total} (diff = {n_total - total})")
    print(f"[A.3 convert] output saved to {out_dir}")
    return 0


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JoyAI-Echo monolithic safetensors to diffusers component layout"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the JoyAI-Echo checkpoint directory (must contain JoyAI-Echo-release.safetensors) "
        "OR the path to the safetensors file directly.",
    )
    parser.add_argument(
        "--gemma-src",
        type=str,
        required=True,
        help="Path to the Gemma text-encoder checkpoint directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=False,
        help="Output directory for the diffusers-style component layout. Required unless --smoke-test.",
    )
    parser.add_argument(
        "--shard-size",
        type=str,
        default="5GB",
        help="Max shard size for save_pretrained (e.g. '5GB', '10GB').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=[None, "source", "fp32", "fp16", "bf16"],
        help="Output dtype. Default: keep source dtype.",
    )
    parser.add_argument(
        "--copy-gemma",
        action="store_true",
        help="Copy Gemma artifacts into the output directory instead of symlinking.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke tests against mock state_dicts (no real conversion) and exit.",
    )
    args = parser.parse_args()
    if not args.smoke_test and not args.out:
        parser.error("--out is required unless --smoke-test is set")
    return args


if __name__ == "__main__":
    sys.exit(main(get_args()))
