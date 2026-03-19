# Word Grid — Architecture & Training Procedure

## Problem Statement

A **word grid** is an N×N grid of English words (N = 5) where:

- Each **row** (read left-to-right) forms a sentence SR_i
- Each **column** (read top-to-bottom) forms a sentence SC_i
- Each sentence is scored on **syntax** (grammar, structure, punctuation, rhythmic flow) from 0.0 to 10.0
- The **grid score** is the average syntax score across all 2N sentences

The goal is to produce the grid with the highest possible score.

### Example (N = 3)

```
Ian    Was   Seen.
Left,  He    Found.
Laredo. Lost? Reborn.
```

| Sentence | Label | Text | Syntax |
|----------|-------|------|--------|
| Row 1 | SR_1 | "Ian Was Seen." | ~8.5 |
| Row 2 | SR_2 | "Left, He Found." | ~7.2 |
| Row 3 | SR_3 | "Laredo. Lost? Reborn." | ~5.0 |
| Col 1 | SC_1 | "Ian Left, Laredo." | ~4.5 |
| Col 2 | SC_2 | "Was He Lost?" | ~9.7 |
| Col 3 | SC_3 | "Seen. Found. Reborn." | ~5.8 |

---

## Architecture

The system follows a **generator–discriminator** architecture trained with reinforcement learning:

```
┌──────────────────── NVIDIA H100 80 GB GPU ──────────────────────┐
│                      TRAINING LOOP (GRPO)                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  GENERATOR   │───▶│ GRID PARSER  │───▶│  DISCRIMINATOR   │   │
│  │  Magistral   │    │              │    │  Mistral-Nemo    │   │
│  │  Small 24B   │    │  Extract     │    │  12B GGUF (GPU)  │   │
│  │ (bf16+LoRA)  │    │  rows + cols │    │  ~7 GB VRAM      │   │
│  │  ~48 GB VRAM │    └──────────────┘    └────────┬─────────┘   │
│  └──────┬───────┘                                 │             │
│         │            ┌──────────────┐             │             │
│         │◀───────────│ GRPO UPDATE  │◀────────────┘             │
│         │            │ (TRL)        │   rewards                 │
│         │            └──────────────┘                           │
│  prompt │                                                       │
│  dataset│    ~67 GB peak / 80 GB total (~13 GB headroom)        │
└─────────┘───────────────────────────────────────────────────────┘
                        ┌──────────────┐
                        │   W&B LOG    │
                        │ + Checkpoint │
                        └──────────────┘
```

### Generator — Magistral-Small-2509

- **Model**: `mistralai/Magistral-Small-2509` (~24 B parameters)
- **Loaded via**: Unsloth `FastLanguageModel` in **full bf16** (not 4-bit quantised)
- **LoRA config**: rank 32, alpha 32, applied to all attention + MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`)
- **Role**: Generates 5×5 word grids as plain text given a themed prompt
- **Runs on**: GPU — ~48 GB VRAM for the frozen bf16 base weights
- **Why bf16 over QLoRA**: Full-precision base weights provide better gradient signal through the LoRA adapters. On the H100 (80 GB) the extra ~36 GB vs. 4-bit is well worth the improved training fidelity.

### Discriminator — Mistral-Nemo-Instruct-2407

- **Model**: `bartowski/Mistral-Nemo-Instruct-2407-GGUF` (Q4_K_M quantisation, 12B params)
- **Loaded via**: `llama-cpp-python` (CPU-only build is sufficient; CUDA build optional)
- **Architecture**: `mistral` — fully supported by `llama-cpp-python ≥ 0.3.x`
- **Role**: Scores individual English sentences on syntax quality (0–10)
- **Runs on**: **GPU** — all layers offloaded (`n_gpu_layers=-1`), ~7 GB VRAM
- **Frozen**: No gradient updates — used only for inference during reward computation
- **Why this model**: The originally intended discriminator (`Ministral-3-8B-Reasoning-2512-GGUF`) uses the `mistral3` GGUF architecture, which is **not supported** by `llama-cpp-python ≤ 0.3.16` (latest release) or `transformers ≤ 4.57`. Mistral-Nemo 12B uses the standard `mistral` architecture, is a stronger scorer than a 7B model, and has official GGUF quantisations available.

---

## Training Procedure

### Algorithm: GRPO (Group Relative Policy Optimization)

We use **GRPO** from TRL rather than PPO because:

1. **No value model** — PPO requires a separate critic network (~24 B extra parameters). With the generator in bf16 (~48 GB) and discriminator on GPU (~7 GB), there is no room for a 24 B critic. GRPO uses within-group reward normalisation instead.
2. **Simple reward integration** — Our custom reward function plugs directly into GRPO's `reward_funcs` interface.
3. **Stable gradients** — Relative advantage estimation within each group of completions reduces variance.

### Training Loop (per step)

1. **Sample prompt** from the dataset (themed instruction to generate a 5×5 grid)
2. **Generate k completions** (default k = 4) using the current policy
3. **Parse each completion** into a `WordGrid` (5 rows × 5 columns)
   - If parsing fails → reward = 0.0 (encourages valid formatting)
4. **Extract 10 sentences** (5 row-sentences + 5 column-sentences) per grid
5. **Score each sentence** with the discriminator → syntax score ∈ [0, 10]
6. **Compute grid reward** = average syntax score / 10 → normalised to [0, 1]
7. **GRPO update**: normalise rewards within the group of k completions, compute policy gradient, update LoRA weights
8. **Log to W&B**: reward mean/std, loss, learning rate, gradient norms
9. **Checkpoint** every `save_steps` steps (rolling window of `save_total_limit`)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base precision | **bf16** | Full-precision base (not 4-bit); native H100 dtype |
| LoRA rank (r) | **32** | Higher rank (was 16) — bf16 headroom allows more expressiveness |
| LoRA alpha | 32 | Standard α = r for unit scaling |
| Learning rate | 5e-6 | Conservative for RL fine-tuning |
| Warmup ratio | 0.1 | Gradual LR ramp-up |
| Batch size | 1 prompt × 4 generations | Fits within ~15 GB KV-cache budget |
| Grad accumulation | 8 | Effective batch = 8 prompts |
| Epochs | 3 | Over 200 prompts = 600 gradient steps |
| Max completion length | 512 tokens | A 5×5 grid ≈ 150–200 tokens + headroom for thinking |
| Discriminator GPU layers | **-1 (all)** | ~7 GB VRAM, 5–10× faster than CPU |

---

## Inputs & Outputs

### Inputs

- **Prompt dataset**: 200 chat-formatted prompts (system + user messages), each requesting a 5×5 word grid with a specific theme and style
- **System prompt**: Instructs the model to output ONLY the grid (one row per line, space-separated words, punctuation attached to words)
- Prompts are generated from 5 templates × 47 themes × 10 styles (up to 2 350 unique combinations)

### Outputs

- **Per grid**: The 5×5 grid text, plus a table of `(sentence, syntax_score)` tuples for all 10 sentences
- **Per grid**: The final grid score (average of all 10 syntax scores)
- **Overall**: The best grid found during evaluation (highest average score)
- **Artefacts**: LoRA adapter weights saved to `checkpoints/final/`, W&B experiment logs

### Output Format Example

```
She    always  found  the    way.
Never  could   see    his    face.
Ran    through dark   cold   night.
The    old     man    saw    light.
Home   was     far    but    near.

  SR_1: 'She always found the way.'              Syntax = 9.2
  SR_2: 'Never could see his face.'              Syntax = 7.8
  SR_3: 'Ran through dark cold night.'           Syntax = 6.5
  SR_4: 'The old man saw light.'                 Syntax = 8.9
  SR_5: 'Home was far but near.'                 Syntax = 8.1
  SC_1: 'She Never Ran The Home'                 Syntax = 3.2
  SC_2: 'always could through old was'           Syntax = 1.1
  SC_3: 'found see dark man far'                 Syntax = 0.8
  SC_4: 'the his cold saw but'                   Syntax = 0.5
  SC_5: 'way. face. night. light. near.'         Syntax = 2.3

  Grid score (avg): 4.84
```

*(This illustrates why the problem is hard — rows are easy, columns require planning.)*

---

## Repository Structure

```
poem/
├── .env.example              # Template for API keys
├── .python-version           # Python 3.11
├── pyproject.toml            # uv/pip dependencies
├── ARCHITECTURE.md           # This document
├── src/
│   └── word_grid/
│       ├── __init__.py       # Package exports
│       ├── config.py         # Environment loading, TrainingConfig dataclass
│       ├── grid.py           # WordGrid class, parsing, sentence extraction
│       ├── discriminator.py  # SentenceScorer (Ministral GGUF via llama-cpp)
│       ├── reward.py         # GridRewardFunction (GRPO-compatible callable)
│       └── prompts.py        # Prompt templates, dataset generation
├── notebooks/
│   └── word_grid_training.ipynb  # Main training notebook (end-to-end)
└── checkpoints/              # Created at runtime — LoRA adapters + optimizer
```

---

## Critiques & Gotchas

### 1. Combinatorial Hardness

A 5×5 grid contains 25 words. Each word must simultaneously serve its row-sentence and its column-sentence. This is a **constraint satisfaction problem** layered on top of language generation. The space of valid grids is vanishingly small compared to all possible 25-word arrangements.

**Mitigation**: An SFT warm-up phase on a curated set of hand-crafted grids would bootstrap the model into the right output space before RL fine-tuning begins.

### 2. Autoregressive Column Blindness

The generator emits tokens left-to-right, row by row. It has no explicit mechanism for ensuring column coherence — that's an emergent property that must be learned. Early in training, rows will score much higher than columns.

**Mitigation**: Consider a structured decoding approach (e.g., column-aware beam search) or a two-pass generation strategy where a draft grid is refined.

### 3. Discriminator Noise & Calibration

The discriminator is an 8 B parameter LLM, not a rule-based grammar checker. Its scores may be:
- **Noisy** across runs (different phrasings of the same quality may get different scores)
- **Biased** toward certain sentence structures
- **Inconsistent** at boundary cases (fragments, unusual punctuation)

**Mitigation**: Average multiple discriminator calls per sentence, or ensemble with a rule-based grammar checker (e.g., `language_tool_python`).

### 4. Reward Sparsity

Until the model learns to produce valid 5-line, 5-words-per-line output, every completion gets reward = 0. This "cold start" can stall GRPO training.

**Mitigation**: Shape the reward — give partial credit for partially valid grids (e.g., 4 out of 5 valid rows). The `partial_bonus` parameter in `GridRewardFunction` supports this.

### 5. GPU Memory Budget

On the H100 (80 GB) with the fully-saturated layout:

| Component | VRAM |
|-----------|------|
| Generator base (bf16 frozen) | ~48 GB |
| LoRA adapters (r=32, 7 modules) | ~400 MB |
| Optimizer states (AdamW, LoRA only) | ~800 MB |
| Activations (gradient checkpointing) | ~5–10 GB |
| KV cache (4 seqs × 512 tokens) | ~1.5 GB |
| Discriminator GGUF (Q4_K_M 12B, all layers) | ~7 GB |
| **Total peak** | **~63–67 GB** |
| **Headroom** | **~13–17 GB** |

This is a tight but comfortable fit. Key things that can push into OOM:
- Increasing `num_generations` beyond 4–6 (more KV cache)
- Increasing `max_completion_length` substantially
- PyTorch's caching allocator holding fragmented memory — call `torch.cuda.empty_cache()` if the discriminator OOMs

### 6. Thinking Token Overhead

Magistral-class models may emit `[THINK]…[/THINK]` reasoning blocks before the actual grid. These tokens consume `max_completion_length` budget without contributing to the grid. The system prompt explicitly asks for "no explanation," but the model may still think internally.

**Mitigation**: Increase `max_completion_length` or strip thinking tokens before length-based truncation.

### 7. Scaling to Larger Grids

For N > 5:
- Completion length grows as O(N²)
- Number of sentences grows as 2N
- Discriminator calls per step grow as 2N × k
- Column coherence becomes exponentially harder

The codebase is parameterised by `grid_size` throughout, so the **code** scales cleanly — but the **problem difficulty** does not.

### 8. GGUF + PyTorch VRAM Contention

Both `llama-cpp-python` and PyTorch allocate GPU memory through separate code paths. PyTorch's caching allocator may hold freed-but-unreturned memory that the GGUF runtime needs, and vice versa.

**Mitigation**:
- Call `torch.cuda.empty_cache()` before discriminator scoring if OOM occurs
- Consider using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- If contention is severe, fall back to `discriminator_n_gpu_layers=0` (CPU) — it's slower but avoids the issue entirely

### 9. llama-cpp-python Must Be Compiled with CUDA

Since the discriminator is fully on GPU (`n_gpu_layers=-1`), `llama-cpp-python` **must** be compiled with CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Without this flag, the GGUF runtime silently falls back to CPU even when `n_gpu_layers=-1`.

### 10. GGUF Architecture Compatibility

Not all Mistral-family models use the same GGUF architecture string. Specifically:

| Model | GGUF `general.architecture` | `llama-cpp-python` 0.3.16 |
|-------|----------------------------|--------------------------|
| Mistral-7B-* | `mistral` | ✅ Supported |
| Mistral-Nemo-* (12B) | `mistral` | ✅ Supported |
| Ministral-3-8B-Reasoning-2512 | `mistral3` | ❌ **Not supported** |

The `mistral3` architecture (used by the Ministral-3 family, released Dec 2025) is too new for any released version of `llama-cpp-python` or `transformers`. The discriminator was switched to **Mistral-Nemo-Instruct-2407** (12B, `mistral` arch) as a result. When `llama-cpp-python` adds `mistral3` support, switching back is a two-line config change.
