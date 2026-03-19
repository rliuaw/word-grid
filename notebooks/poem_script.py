import modal
app = modal.App("poem")

# Use NVIDIA CUDA devel image as the base so nvcc is available for
# compiling llama-cpp-python with GPU support.  The devel image ships
# the full CUDA toolkit (headers, nvcc, libraries) — not just the driver.
train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.46.0,<5.0",
        "trl>=0.24.0,<0.25",
        "peft>=0.13.0",
        "datasets>=3.1.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.44.0",
        "unsloth==2026.2.1",
        "unsloth_zoo==2026.2.1",
        "xformers",
        "wandb>=0.18.0",
        "python-dotenv>=1.0.0",
        "huggingface-hub>=0.26.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.0.0",
    )
    # Build llama-cpp-python from source with CUDA.  nvcc comes from
    # the nvidia/cuda devel base; gcc/g++ from build-essential.
    # Override CC/CXX (the base image sets them to clang which isn't installed).
    .apt_install("build-essential")
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands(
        "CMAKE_ARGS='-DGGML_CUDA=on' pip install 'llama-cpp-python>=0.3.0' "
        "--no-binary llama-cpp-python",
    )
    # CRITICAL: Force unbuffered stdout so logs appear in real-time on Modal
    .env({"HF_HOME": "/model_cache", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("word_grid")
)

with train_image.imports():
    # unsloth must be first!
    import unsloth  # noqa: F401,I001

    import logging
    import os
    import sys
    import time

    import torch
    import wandb
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel

    from word_grid.config import (
        GENERATOR_MODEL,
        HF_TOKEN,
        WANDB_API_KEY,
        TrainingConfig,
    )
    from word_grid.discriminator import SentenceScorer
    from word_grid.grid import parse_grid_from_text, score_report
    from word_grid.prompts import build_prompt_dataset, build_prompt_messages
    from word_grid.reward import GridRewardFunction

model_cache_volume = modal.Volume.from_name(
    "unsloth-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "unsloth-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "unsloth-checkpoints", create_if_missing=True
)

GPU_TYPE = "H100"
TIMEOUT_MINUTES = 6*60
MAX_RETRIES = 3
NUM_PROMPTS = 200
NUM_EVAL = 20


# ---------------------------------------------------------------------------
# Helpers (must not reference train_image.imports() names at module level)
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Print with flush to ensure immediate visibility on Modal."""
    print(msg, flush=True)


def timed(phase_name: str):
    """Context manager that logs elapsed wall-clock time for a phase."""
    import contextlib
    import time as _time

    @contextlib.contextmanager
    def _ctx():
        log(f"⏱  [{phase_name}] start")
        t0 = _time.time()
        yield
        elapsed = _time.time() - t0
        log(f"⏱  [{phase_name}] done — {elapsed:.1f}s")
    return _ctx()

@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    cpu=1.5,
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=TIMEOUT_MINUTES * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    single_use_containers=True,
)
def finetune():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)
    run_t0 = time.time()
    _wandb = wandb  # alias for use in callbacks

    # ── Proposal A: Heartbeat background thread ────────────────────
    import threading

    _heartbeat_stop = threading.Event()

    def _heartbeat_loop(interval: int = 120):
        """Print a heartbeat every *interval* seconds so Modal/logs
        never appear stalled during long discriminator scoring loops."""
        n = 0
        while not _heartbeat_stop.wait(interval):
            n += 1
            alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            log(
                f"💓  heartbeat #{n} — alive, "
                f"VRAM alloc={alloc:.1f}G reserved={reserved:.1f}G"
            )

    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    log("Heartbeat thread started (120 s interval)")

    # ── Callbacks (defined here so TrainerCallback import is available) ──
    class VolumePersistCallback(TrainerCallback):
        """Commit the checkpoint volume after every HF Trainer save event."""

        def __init__(self, volume, name: str = "checkpoints"):
            self.volume = volume
            self.name = name

        def on_save(self, args, state, control, **kwargs):
            log(f"💾  Persisting {self.name} volume (step {state.global_step})…")
            self.volume.commit()
            log(f"💾  {self.name} volume committed ✓")

    # ── Proposal B: Step-boundary timing callback ────────────────
    class StepTimingCallback(TrainerCallback):
        """Log wall-clock time per training step + VRAM usage."""

        def __init__(self):
            self._step_t0 = None

        def on_step_begin(self, args, state, control, **kwargs):
            self._step_t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if self._step_t0 is not None:
                elapsed = time.time() - self._step_t0
                alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                log(
                    f"📊  step {state.global_step}: {elapsed:.1f}s, "
                    f"VRAM={alloc:.1f}G, loss={state.log_history[-1].get('loss', '?') if state.log_history else '?'}"
                )
                if _wandb is not None and _wandb.run is not None:
                    _wandb.log({
                        "perf/step_time_s": elapsed,
                        "perf/vram_alloc_gb": alloc,
                    }, commit=False)

    # ── Configuration ────────────────────────────────────────────
    cfg = TrainingConfig()

    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU:             {torch.cuda.get_device_name(0)}")
        log(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)

    log("=== Training Configuration ===")
    log(f"  Grid size:        {cfg.grid_size}×{cfg.grid_size}")
    log(f"  Generator:        {GENERATOR_MODEL}")
    log(f"  Load in 4-bit:    {cfg.load_in_4bit}")
    log(f"  LoRA rank:        {cfg.lora_r}")
    log(f"  LoRA alpha:       {cfg.lora_alpha}")
    log(f"  Learning rate:    {cfg.learning_rate}")
    log(f"  Epochs:           {cfg.num_train_epochs}")
    log(f"  Batch size:       {cfg.per_device_train_batch_size} prompt × {cfg.num_generations} gens")
    log(f"  Grad accum:       {cfg.gradient_accumulation_steps}")
    log(f"  Disc GPU layers:  {cfg.discriminator_n_gpu_layers}")
    log(f"  Disc n_ctx:       {cfg.discriminator_n_ctx}")
    log(f"  Disc max_tokens:  {cfg.discriminator_max_tokens}")
    log(f"  Max compl. len:   {cfg.max_completion_length}")

    # ── Step 1: Load discriminator ───────────────────────────────
    with timed("Load discriminator"):
        scorer = SentenceScorer(
            n_ctx=cfg.discriminator_n_ctx,
            n_gpu_layers=cfg.discriminator_n_gpu_layers,
            temperature=cfg.discriminator_temperature,
            max_tokens=cfg.discriminator_max_tokens,
        )
    log("Discriminator loaded ✓")

    # Verify discriminator is using GPU
    if cfg.discriminator_n_gpu_layers == 0:
        raise RuntimeError(
            "Discriminator is configured with n_gpu_layers=0 (CPU-only). "
            "Set discriminator_n_gpu_layers to -1 to offload all layers to GPU."
        )
    # llama-cpp-python exposes n_gpu_layers as a C-struct attribute (not a dict)
    disc_gpu_layers = getattr(scorer.llm.model_params, "n_gpu_layers", None) if hasattr(scorer.llm, "model_params") else None
    if disc_gpu_layers is not None and disc_gpu_layers == 0:
        raise RuntimeError(
            "Discriminator reports 0 GPU-offloaded layers at runtime. "
            "Ensure CUDA is available and llama-cpp-python was built with GPU support."
        )
    log(f"  Discriminator GPU layers: {cfg.discriminator_n_gpu_layers} (requested)")

    # Warmup: first inference call triggers JIT / CUDA kernel compilation
    with timed("Discriminator warmup"):
        warmup_score = scorer.score_sentence("The quick brown fox jumps.")
        log(f"  Warmup score: {warmup_score:.1f}")

    # Persist model downloads to the volume so retries skip re-downloading
    model_cache_volume.commit()
    log("Model cache committed ✓")

    # ── Step 2: Load generator with LoRA ─────────────────────────
    with timed("Load generator"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=GENERATOR_MODEL,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=torch.bfloat16 if cfg.bf16 else None,
            token=HF_TOKEN,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            use_gradient_checkpointing="unsloth",
        )

    trainable, total = model.get_nb_trainable_parameters()
    log("Generator loaded ✓")

    # Verify generator is on GPU
    gen_device = next(model.parameters()).device
    if gen_device.type != "cuda":
        raise RuntimeError(
            f"Generator model is on {gen_device}, expected CUDA GPU. "
            "Ensure a GPU is available and the model is loaded correctly."
        )
    log(f"  Generator device: {gen_device}")

    log(f"  Total params:     {total:>14,}")
    log(f"  Trainable params: {trainable:>14,} ({100 * trainable / total:.2f}%)")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log(f"  VRAM allocated:   {alloc:.1f} GB")
        log(f"  VRAM reserved:    {reserved:.1f} GB")

    # Persist generator weights to the volume
    model_cache_volume.commit()
    log("Generator cache committed ✓")

    # ── Step 3: Build prompt dataset ─────────────────────────────
    # uniform=True is required to work around an Unsloth 2026.2.1 bug
    # where variable-length prompt left-padding causes a tensor shape
    # mismatch in compute_loss.  The seed controls which theme/style/
    # template combo is sampled, so varying the seed across runs gives
    # prompt diversity without triggering the bug.
    dataset = build_prompt_dataset(
        n=cfg.grid_size, num_prompts=NUM_PROMPTS, seed=42, uniform=True,
    )
    log(f"Dataset size: {len(dataset)} prompts")

    # ── Step 4: Create reward function (with instrumentation) ────
    reward_fn = GridRewardFunction(
        scorer=scorer,
        grid_size=cfg.grid_size,
        invalid_reward=cfg.invalid_grid_reward,
        partial_bonus=cfg.partial_grid_bonus,
    )

    # ── Step 5: GRPO training ────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        # ── Performance: explicit generation temperature for GRPO
        temperature=0.9,
        # ── Instrumentation: log sample completions to W&B
        log_completions=True,
        # ── Performance: use static KV cache for faster generation
        cache_implementation="static",
    )

    # Attach callbacks
    persist_cb = VolumePersistCallback(checkpoint_volume, name="checkpoints")
    timing_cb = StepTimingCallback()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        callbacks=[persist_cb, timing_cb],
    )

    steps_per_epoch = len(dataset) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    log("GRPOTrainer initialised ✓")
    log(f"  Steps/epoch:  ~{steps_per_epoch}")
    log(f"  Total epochs:  {cfg.num_train_epochs}")
    log(f"  Total steps:  ~{steps_per_epoch * cfg.num_train_epochs}")
    log(f"\nStarting training…")

    with timed("GRPO training"):
        trainer.train()
    log("\n✓ Training complete")

    # ── Step 6: Save model ───────────────────────────────────────
    save_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    checkpoint_volume.commit()
    log(f"LoRA adapters + tokenizer saved to {save_path} ✓")

    # ── Step 7: Evaluation ───────────────────────────────────────
    with timed("Evaluation"):
        FastLanguageModel.for_inference(model)

        best_reward = -1.0
        best_grid = None
        best_scores = None
        results = []
        parse_failures = 0

        for i in range(NUM_EVAL):
            messages = build_prompt_messages(n=cfg.grid_size)
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs,
                    max_new_tokens=cfg.max_completion_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            completion = tokenizer.decode(
                output_ids[0][inputs.shape[-1]:],
                skip_special_tokens=True,
            )

            grid = parse_grid_from_text(completion, n=cfg.grid_size)
            if grid is not None:
                avg, scores = reward_fn.evaluate_grid(grid)
                results.append((avg, grid, scores))
                if avg > best_reward:
                    best_reward = avg
                    best_grid = grid
                    best_scores = scores
                log(f"  Grid {i+1:2d}: avg syntax = {avg:.2f}")
            else:
                parse_failures += 1
                log(f"  Grid {i+1:2d}: PARSE FAILED")

    log(f"\nGenerated {len(results)}/{NUM_EVAL} valid grids ({parse_failures} parse failures)")

    # Log evaluation summary to W&B
    if wandb.run is not None and results:
        eval_rewards = [r[0] for r in results]
        wandb.log({
            "eval/num_valid_grids": len(results),
            "eval/num_parse_failures": parse_failures,
            "eval/best_reward": best_reward,
            "eval/mean_reward": sum(eval_rewards) / len(eval_rewards),
        })

    if best_grid is not None:
        log("=" * 60)
        log("  BEST GRID")
        log("=" * 60)
        log(score_report(best_grid, best_scores))
        log("=" * 60)
    else:
        log("No valid grids were generated.")

    # ── Step 8: Cleanup ──────────────────────────────────────────
    _heartbeat_stop.set()  # stop heartbeat thread
    total_elapsed = time.time() - run_t0
    log(f"\n⏱  Total wall-clock time: {total_elapsed / 60:.1f} min")

    if wandb.run is not None:
        wandb.finish()
        log("W&B run finished ✓")

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    log("GPU memory released ✓")

    return cfg.run_name

@app.local_entrypoint()
def main():
    print("Launching word-grid GRPO training on Modal…")
    print(f"  GPU: {GPU_TYPE}")
    print(f"  Timeout: {TIMEOUT_MINUTES} min")
    print(f"  Prompts: {NUM_PROMPTS}")
    print(f"  Eval grids: {NUM_EVAL}")

    # Launch the training job on Modal infrastructure
    run_name = finetune.remote()
    print(f"\n✓ Training completed successfully: {run_name}")

