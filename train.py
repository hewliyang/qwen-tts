"""
SFT training for Qwen3-TTS on MLX (Apple Silicon).

Uses LoRA adapters on the talker transformer to learn voice
characteristics without destroying speech coherence. Only a small
fraction of parameters are trained (~2-4M vs 755M frozen).

Usage:
    python train.py \\
        --train-jsonl train.jsonl \\
        --speaker-name lky \\
        --output ./voices/lky/ \\
        --epochs 3 \\
        --lr 2e-5
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_audio.tts.models.qwen3_tts import Model as Qwen3TTSModel
from mlx_lm.tuner.utils import LoRALinear

from dataset import Batch, TTSDataset, load_jsonl


def _flatten_arrays(
    tree: object,
) -> list[tuple[str, mx.array]]:
    """Flatten a parameter tree, casting values to mx.array."""
    return [(k, cast(mx.array, v)) for k, v in tree_flatten(tree)]


def _add_flat_grads(
    flat_a: list[tuple[str, mx.array]],
    flat_b: list[tuple[str, mx.array]],
) -> list[tuple[str, mx.array]]:
    """Element-wise add two flattened gradient lists."""
    return [(k1, v1 + v2) for (k1, v1), (_k2, v2) in zip(flat_a, flat_b, strict=False)]


def _scale_flat_grads(
    flat: list[tuple[str, mx.array]],
    scale: float,
) -> list[tuple[str, mx.array]]:
    """Scale flattened gradients by a scalar."""
    return [(k, v * scale) for k, v in flat]


def _flat_grads_to_tree(flat: list[tuple[str, mx.array]]) -> object:
    """Convert flattened gradients back to tree."""
    return tree_unflatten(flat)


def _global_grad_norm(flat_grads: list[tuple[str, mx.array]]) -> mx.array:
    """Compute global L2 grad norm from flattened gradients."""
    sq_sum = mx.array(0.0)
    for _, g in flat_grads:
        sq_sum = sq_sum + (g * g).sum()
    return mx.sqrt(sq_sum)


def cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = -100,
) -> mx.array:
    """Compute cross-entropy loss, ignoring ignore_index."""
    mask = mx.array(labels != ignore_index)
    valid_count = mask.sum()

    if valid_count == 0:
        return mx.array(0.0)

    # Gather log-probs at label positions
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)

    # Replace ignore_index with 0 for gathering
    safe_labels = mx.where(flat_labels == ignore_index, 0, flat_labels)

    # Cross entropy: -log_softmax(logits)[label]
    log_probs = flat_logits - mx.logsumexp(flat_logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(log_probs, safe_labels[:, None], axis=-1).squeeze(-1)

    # Mask and average
    masked_nll = nll * mask.reshape(-1).astype(nll.dtype)
    return masked_nll.sum() / valid_count


def forward_sub_talker(
    model: Qwen3TTSModel,
    codec_ids_masked: mx.array,
    hidden_states_masked: mx.array,
) -> mx.array:
    """Forward pass through the sub-talker (code predictor).

    Replicates the PyTorch forward_sub_talker_finetune logic:
    - Build input embeddings:
      [hidden_state, codec_embed(code_0), ...]
    - Forward through code predictor model
    - Compute per-group logits and cross-entropy loss

    Args:
        model: The loaded Qwen3-TTS model
        codec_ids_masked: [N, 16] codec IDs at codec positions
        hidden_states_masked: [N, hidden_dim] hidden states

    Returns:
        sub_talker_loss: scalar loss
    """
    talker = model.talker
    code_predictor = talker.code_predictor
    talker_cfg = model.config.talker_config
    if talker_cfg is None:
        raise ValueError("Missing talker_config in model config")
    num_code_groups = talker_cfg.num_code_groups

    # Position 0: hidden states from talker
    sub_inputs = [hidden_states_masked[:, None, :]]

    # Positions 1..15: codec embeddings
    for code_idx in range(num_code_groups - 1):
        if code_idx == 0:
            emb = talker.get_input_embeddings()(codec_ids_masked[:, :1])
        else:
            cb_emb_layer = code_predictor.codec_embedding[code_idx - 1]
            sl = slice(code_idx, code_idx + 1)
            emb = cb_emb_layer(codec_ids_masked[:, sl])
        sub_inputs.append(emb)

    sub_inputs_embeds = mx.concatenate(sub_inputs, axis=1)

    # Project if needed
    proj = code_predictor.small_to_mtp_projection
    if proj is not None:
        sub_inputs_embeds = proj(sub_inputs_embeds)

    # Forward through code predictor transformer
    hidden = code_predictor.model(sub_inputs_embeds)

    # Compute per-group logits and loss
    labels = codec_ids_masked[:, 1:]  # [N, 15]

    total_loss = mx.array(0.0)
    for group_idx in range(num_code_groups - 1):
        head = code_predictor.lm_head[group_idx]
        group_logits = head(hidden[:, group_idx + 1, :])
        group_labels = labels[:, group_idx]
        group_loss = cross_entropy_loss(
            group_logits[None, :, :],
            group_labels[None, :],
        )
        total_loss = total_loss + group_loss

    return total_loss / (num_code_groups - 1)


def train_step(model: Qwen3TTSModel, batch: Batch) -> mx.array:
    """Single training step: forward pass computing total loss.

    Replicates the PyTorch training logic from sft_12hz.py:
    1. Extract speaker embedding from ref_mels (frozen)
    2. Build input embeddings (text + codec + speaker +
       sub-codebooks)
    3. Forward talker -> logits -> CE loss
    4. Forward sub-talker -> logits per code group -> CE loss
    5. total_loss = talker_loss + 0.3 * sub_talker_loss
    """
    input_ids = batch.input_ids  # [B, T, 2]
    codec_ids = batch.codec_ids  # [B, T, 16]
    ref_mels = batch.ref_mels  # [B, frames, 128]
    text_embedding_mask = batch.text_embedding_mask
    codec_embedding_mask = batch.codec_embedding_mask
    codec_0_labels = batch.codec_0_labels  # [B, T]
    codec_mask = batch.codec_mask  # [B, T]

    # 1. Speaker embedding (frozen)
    if model.speaker_encoder is None:
        raise ValueError("Model is missing speaker_encoder")
    speaker_embedding = mx.stop_gradient(model.speaker_encoder(ref_mels))

    # 2. Build input embeddings
    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    # Text embeddings
    talker_cfg = model.config.talker_config
    if talker_cfg is None:
        raise ValueError("Missing talker_config in model config")
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if talker_cfg.text_hidden_size != talker_cfg.hidden_size:
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = (
        model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    )

    # Inject speaker embedding at position 6
    B = input_codec_embedding.shape[0]
    input_codec_embedding = mx.concatenate(
        [
            input_codec_embedding[:, :6, :],
            speaker_embedding[:, None, :],
            input_codec_embedding[:, 7:, :],
        ],
        axis=1,
    )

    input_embeddings = input_text_embedding + input_codec_embedding

    # Add sub-codebook embeddings (codebooks 1-15)
    codec_mask_3d = codec_mask[:, :, None]
    for cb_idx in range(1, 16):
        cb_emb_layer = model.talker.code_predictor.codec_embedding[cb_idx - 1]
        cb_embedding = cb_emb_layer(codec_ids[:, :, cb_idx])
        cb_embedding = cb_embedding * codec_mask_3d
        input_embeddings = input_embeddings + cb_embedding

    # 3. Forward through talker (shifted by 1)
    talker_logits, talker_hidden = model.talker(
        input_embeddings[:, :-1, :],
    )

    # Talker loss (CE on codec_0)
    talker_loss = cross_entropy_loss(talker_logits, codec_0_labels[:, 1:])

    # 4. Sub-talker loss
    shifted_codec_mask = codec_mask[:, 1:]

    sub_talker_loss = mx.array(0.0)
    for batch_idx in range(B):
        mask_b = shifted_codec_mask[batch_idx]

        # Convert to numpy for index computation
        mask_np = np.array(mask_b)
        shifted_indices = np.nonzero(mask_np)[0]
        if shifted_indices.size == 0:
            continue
        shifted_indices_mx = mx.array(shifted_indices)

        hidden_b = talker_hidden[batch_idx]
        hidden_masked = hidden_b[shifted_indices_mx]

        # Original (unshifted) codec positions corresponding to shifted mask.
        # shifted_codec_mask = codec_mask[:, 1:], so indices map by +1.
        orig_indices_mx = mx.array(shifted_indices + 1)
        codec_masked = codec_ids[batch_idx][orig_indices_mx]

        sub_loss_b = forward_sub_talker(model, codec_masked, hidden_masked)
        sub_talker_loss = sub_talker_loss + sub_loss_b

    sub_talker_loss = sub_talker_loss / B

    # 5. Total loss
    total_loss = talker_loss + 0.3 * sub_talker_loss
    return total_loss


def apply_lora_to_talker(
    model: Qwen3TTSModel,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_layers: int | None = None,
) -> tuple[int, int]:
    """Apply LoRA adapters to the talker attention layers.

    Freezes the entire model first, then replaces attention
    projections with LoRA variants (which are trainable).

    Returns:
        (trainable_params, total_params)
    """
    # 1. Freeze everything
    model.freeze()

    # 2. Apply LoRA to talker attention layers
    talker_layers = model.talker.model.layers
    num_layers = len(talker_layers)
    if lora_layers is None:
        lora_layers = num_layers

    target_keys = {"self_attn.q_proj", "self_attn.v_proj"}
    start_layer = max(0, num_layers - lora_layers)

    for i in range(start_layer, num_layers):
        layer = talker_layers[i]
        lora_modules = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(
                    module,
                    r=lora_rank,
                    scale=lora_scale,
                    dropout=0.0,
                )
                lora_modules.append((key, lora_layer))
        if lora_modules:
            layer.update_modules(tree_unflatten(lora_modules))

    # 3. Also unfreeze codec_head and text_projection
    model.talker.codec_head.unfreeze()
    if hasattr(model.talker, "text_projection"):
        model.talker.text_projection.unfreeze()

    # Count trainable params
    trainable_params = model.trainable_parameters()
    all_params = model.parameters()
    trainable = sum(v.size for _, v in _flatten_arrays(trainable_params))
    total = sum(v.size for _, v in _flatten_arrays(all_params))
    return trainable, total


def save_checkpoint(
    model: Qwen3TTSModel,
    model_path: Path,
    output_dir: str,
    speaker_name: str,
    speaker_embedding: mx.array,
) -> None:
    """Save fine-tuned model as a CustomVoice checkpoint.

    Merges LoRA weights mathematically (W + scale * B @ A)
    without modifying the live model, so training can continue.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy base model files
    print(f"  Copying base model from {model_path}...")
    shutil.copytree(str(model_path), str(output_path), dirs_exist_ok=True)

    # Update config.json
    config_path = output_path / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Collect model weights, merging LoRA on the fly
    print("  Collecting & merging weights...")
    weights = dict(tree_flatten(model.parameters()))

    lora_keys = [k for k in weights if k.endswith(".lora_a")]
    merged_count = 0
    for lora_a_key in lora_keys:
        base_prefix = lora_a_key.rsplit(".lora_a", 1)[0]
        lora_b_key = base_prefix + ".lora_b"
        weight_key = base_prefix + ".linear.weight"

        if lora_b_key in weights and weight_key in weights:
            lora_a = weights[lora_a_key]
            lora_b = weights[lora_b_key]

            # Get scale from the LoRA module
            parts = base_prefix.split(".")
            mod = model
            for p in parts:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            scale = mod.scale if hasattr(mod, "scale") else 20.0

            # Merge: W_new = W + scale * B.T @ A.T
            delta = (scale * lora_b.T) @ lora_a.T
            w = weights[weight_key]
            weights[weight_key] = w + delta.astype(w.dtype)

            # Rename weight key (remove .linear.)
            orig_weight_key = base_prefix + ".weight"
            weights[orig_weight_key] = weights.pop(weight_key)

            # Also rename .linear.bias if present
            linear_bias_key = base_prefix + ".linear.bias"
            if linear_bias_key in weights:
                weights[base_prefix + ".bias"] = weights.pop(linear_bias_key)

            merged_count += 1

    if merged_count:
        print(f"  Merged {merged_count} LoRA adapters")

    # Drop LoRA keys, speaker_encoder, speech_tokenizer
    keys_to_drop = [k for k in weights if any(x in k for x in [".lora_a", ".lora_b"])]
    keys_to_drop += [
        k for k in weights if k.startswith(("speaker_encoder", "speech_tokenizer"))
    ]
    for k in set(keys_to_drop):
        weights.pop(k, None)

    # Burn speaker embedding into codec_embedding.weight[3000]
    codec_key = "talker.model.codec_embedding.weight"
    if codec_key in weights:
        codec_weight = weights[codec_key]
        spk_emb = speaker_embedding.reshape(-1)
        before = codec_weight[:3000]
        after = codec_weight[3001:]
        weights[codec_key] = mx.concatenate([before, spk_emb[None, :], after], axis=0)

    # Save weights
    print("  Saving model weights...")
    for sf in output_path.glob("model*.safetensors"):
        sf.unlink()

    mx.save_safetensors(str(output_path / "model.safetensors"), weights)
    print(f"  ✅ Checkpoint saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for Qwen3-TTS on MLX")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="Training JSONL (output of prepare_data.py)",
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        default="speaker",
        help="Speaker name for the fine-tuned model",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./voice_output",
        help="Output directory for checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"),
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (1 recommended for memory)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save checkpoint after every epoch",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print optimizer-step loss every N updates (0 disables step logs)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (0 = full fine-tune)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=20.0,
        help="LoRA scale/alpha",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=None,
        help="Number of transformer layers for LoRA",
    )
    args = parser.parse_args()

    print("🎙️  Qwen3-TTS SFT Training (MLX) — LoRA")
    print(f"{'=' * 50}")
    print(f"  Model:        {args.model}")
    print(f"  Speaker:      {args.speaker_name}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Grad accum:   {args.grad_accum}")
    print(f"  Log every:    {args.log_every}")
    print(f"  LoRA rank:    {args.lora_rank}")
    print(f"  LoRA scale:   {args.lora_scale}")
    print(f"  LoRA layers:  {args.lora_layers or 'all'}")
    print(f"  Output:       {args.output}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    t0 = time.time()
    from mlx_audio.tts.utils import load_model
    from mlx_audio.utils import get_model_path

    model_path = get_model_path(args.model)
    model = cast(Qwen3TTSModel, load_model(model_path))
    print(f"done ({time.time() - t0:.1f}s)")

    if model.speaker_encoder is None:
        raise ValueError(
            "The model does not have a speaker encoder. "
            "Fine-tuning requires a base model with "
            "speaker encoder."
        )

    # Apply LoRA
    if args.lora_rank > 0:
        print("Applying LoRA...", end=" ", flush=True)
        trainable, total = apply_lora_to_talker(
            model,
            lora_rank=args.lora_rank,
            lora_scale=args.lora_scale,
            lora_layers=args.lora_layers,
        )
        print(f"done ({trainable / 1e6:.1f}M / {total / 1e6:.1f}M trainable)")
    else:
        print("⚠️  Full fine-tune mode (no LoRA) — may degrade speech quality!")

    # Load training data
    print("Loading training data...", end=" ", flush=True)
    train_data = load_jsonl(args.train_jsonl)
    print(f"{len(train_data)} samples")

    dataset = TTSDataset(train_data, model.tokenizer, model.config)

    # Extract speaker embedding from first sample
    print("[1/4] Loading ref audio...", flush=True)
    from dataset import extract_mels, load_audio_24k

    ref_audio_path = train_data[0].ref_audio
    ref_audio, ref_sr = load_audio_24k(ref_audio_path)
    print(
        f"  audio: {len(ref_audio) / ref_sr:.1f}s",
        flush=True,
    )
    print("[2/4] Computing mel...", flush=True)
    ref_mel = extract_mels(ref_audio, ref_sr)
    mx.eval(ref_mel)
    print(f"  mel: {ref_mel.shape}", flush=True)
    print("[3/4] Speaker encoder...", flush=True)
    target_speaker_embedding = model.speaker_encoder(ref_mel)
    print("  forward done, evaluating...", flush=True)
    mx.eval(target_speaker_embedding)
    print(
        f"[4/4] Done: {target_speaker_embedding.shape}",
        flush=True,
    )

    # Setup optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)

    # Build loss+grad function
    loss_and_grad_fn = nn.value_and_grad(model, train_step)

    # Training loop
    print(f"\n{'=' * 50}")
    print("Starting training...")
    print(f"{'=' * 50}\n", flush=True)

    accum_loss_mx = mx.array(0.0)
    accum_steps = 0
    accumulated_flat_grads: list[tuple[str, mx.array]] | None = None
    optimizer_steps = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_steps = 0

        # Shuffle indices
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        # Process in batches
        for batch_start in range(0, len(indices), args.batch_size):
            batch_end = batch_start + args.batch_size
            batch_indices = indices[batch_start:batch_end]
            batch_items = [dataset[i] for i in batch_indices]
            batch = dataset.collate(batch_items)

            # Forward + backward
            loss, grads = loss_and_grad_fn(model, batch)

            # Gradient accumulation
            flat_new = _flatten_arrays(grads)
            if accumulated_flat_grads is None:
                accumulated_flat_grads = flat_new
            else:
                accumulated_flat_grads = _add_flat_grads(
                    accumulated_flat_grads,
                    flat_new,
                )

            accum_steps += 1
            accum_loss_mx = accum_loss_mx + loss

            if accum_steps >= args.grad_accum:
                # Average gradients
                accumulated_flat_grads = _scale_flat_grads(
                    accumulated_flat_grads,
                    1.0 / args.grad_accum,
                )

                # Gradient clipping
                if args.max_grad_norm > 0:
                    total_norm = _global_grad_norm(accumulated_flat_grads)
                    mx.eval(total_norm)
                    clip_coef = args.max_grad_norm / (total_norm.item() + 1e-6)
                    if clip_coef < 1.0:
                        accumulated_flat_grads = _scale_flat_grads(
                            accumulated_flat_grads,
                            clip_coef,
                        )

                # Update
                accumulated_grads = _flat_grads_to_tree(accumulated_flat_grads)
                optimizer.update(model, cast(dict, accumulated_grads))

                avg_loss_mx = accum_loss_mx / accum_steps
                mx.eval(avg_loss_mx, model.parameters(), optimizer.state)
                avg_loss = float(avg_loss_mx.item())

                optimizer_steps += 1
                if args.log_every > 0 and (optimizer_steps % args.log_every == 0):
                    step_idx = batch_start // args.batch_size
                    print(f"  Epoch {epoch} | Step {step_idx} | Loss: {avg_loss:.4f}")

                epoch_loss += avg_loss
                num_steps += 1
                accum_loss_mx = mx.array(0.0)
                accum_steps = 0
                accumulated_flat_grads = None

        # Handle remaining accumulated gradients
        if accum_steps > 0 and accumulated_flat_grads is not None:
            accumulated_flat_grads = _scale_flat_grads(
                accumulated_flat_grads,
                1.0 / accum_steps,
            )
            accumulated_grads = _flat_grads_to_tree(accumulated_flat_grads)
            optimizer.update(model, cast(dict, accumulated_grads))
            avg_loss_mx = accum_loss_mx / accum_steps
            mx.eval(avg_loss_mx, model.parameters(), optimizer.state)
            epoch_loss += float(avg_loss_mx.item())
            num_steps += 1
            accum_loss_mx = mx.array(0.0)
            accum_steps = 0
            accumulated_flat_grads = None

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_steps, 1)
        print(
            f"\n  Epoch {epoch} complete | "
            f"Avg Loss: {avg_epoch_loss:.4f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

        mx.clear_cache()

        # Save checkpoint
        if args.save_every_epoch or epoch == args.epochs - 1:
            if args.save_every_epoch:
                ckpt_dir = os.path.join(
                    args.output,
                    f"checkpoint-epoch-{epoch}",
                )
            else:
                ckpt_dir = args.output
            save_checkpoint(
                model,
                model_path,
                ckpt_dir,
                args.speaker_name,
                target_speaker_embedding,
            )

    print(f"\n{'=' * 50}")
    print("✅ Training complete!")
    print(f"{'=' * 50}")
    print("\nTo generate with your fine-tuned voice:")
    print(
        f'  qwen-tts generate -p "Hello world" '
        f"--speaker {args.speaker_name} "
        f"--voice-model {args.output}"
    )


if __name__ == "__main__":
    main()
