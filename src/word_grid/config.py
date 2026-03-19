"""Configuration constants and environment loading for word-grid training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (two levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# Full-precision bf16 model — *not* the bnb-4bit variant.  On an H100 80 GB
# the 24 B-param model fits comfortably in bf16 (~48 GB) with LoRA, leaving
# room for the discriminator and training dynamics.
GENERATOR_MODEL: str = os.getenv(
    "GENERATOR_MODEL",
    "unsloth/Magistral-Small-2509",
)
GENERATOR_MODEL_FULL: str = "mistralai/Magistral-Small-2509"

DISCRIMINATOR_REPO: str = os.getenv(
    "DISCRIMINATOR_MODEL",
    "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
)
DISCRIMINATOR_GGUF_FILE: str = os.getenv(
    "DISCRIMINATOR_GGUF_FILE",
    "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
)

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
WANDB_API_KEY: str | None = os.getenv("WANDB_API_KEY")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "word-grid")
WANDB_ENTITY: str | None = os.getenv("WANDB_ENTITY") or None

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
# DEFAULT_GRID_SIZE: int = 5
DEFAULT_GRID_SIZE: int = 3
MAX_SEQ_LENGTH: int = 2048
MAX_COMPLETION_LENGTH: int = 1024


@dataclass
class TrainingConfig:
    """All tuneable hyper-parameters in one place.

    Designed to saturate an H100 80 GB GPU:
      • Generator in full bf16 (LoRA, *not* QLoRA) ≈ 48 GB
      • Discriminator GGUF fully on GPU              ≈  7 GB
      • Training dynamics (optimizer, activations)   ≈ 12 GB
      • Headroom                                     ≈ 15 GB
    """

    grid_size: int = DEFAULT_GRID_SIZE

    # Generator precision — False = full bf16, True = 4-bit QLoRA
    load_in_4bit: bool = False

    # LoRA (full-rank bf16 base; only adapters are trainable)
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # GRPO
    num_generations: int = 2   # fewer gens → 2× fewer gen + disc calls per step
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # 4 (not 8) → halves first-step latency
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    max_completion_length: int = MAX_COMPLETION_LENGTH
    max_seq_length: int = MAX_SEQ_LENGTH
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging / checkpointing
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 5
    output_dir: str = "./checkpoints"
    report_to: str = "wandb"
    run_name: str = "word-grid-grpo"

    # Discriminator — fully on GPU for fast scoring
    discriminator_n_ctx: int = 512   # prompts are <100 tokens; saves memory + KV alloc
    discriminator_n_gpu_layers: int = -1  # -1 = ALL layers on GPU
    discriminator_temperature: float = 0.0
    discriminator_max_tokens: int = 30  # response is ~15 tokens: {"syntax_score": 7.5}

    # Reward shaping
    invalid_grid_reward: float = 0.0
    partial_grid_bonus: float = 0.1
