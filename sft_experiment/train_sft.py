#!/usr/bin/env python
"""Small, config-driven SFT runner for smoke tests and the Olmo 3 experiment.

The smoke path loads the real Olmo 3 tokenizer/config, shrinks the architecture,
and still exercises TRL's SFTTrainer and the evaluation metrics used by the
experiment. The full path loads the configured pretrained model unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-kl", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def label_smoothing_tag(value: float) -> str:
    return f"ls{int(round(value * 100)):03d}"


def build_smoke_config(base_config, overrides: dict[str, Any]):
    config = copy.deepcopy(base_config)
    for name, value in overrides.items():
        if hasattr(config, name):
            setattr(config, name, value)

    # Keep grouped-query attention internally valid after shrinking the model.
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = min(config.num_attention_heads, 8)
        config.num_attention_heads = max(config.num_attention_heads, 1)
    if hasattr(config, "num_key_value_heads"):
        config.num_key_value_heads = min(
            config.num_key_value_heads, config.num_attention_heads
        )
        while config.num_attention_heads % config.num_key_value_heads != 0:
            config.num_key_value_heads -= 1
    if hasattr(config, "vocab_size"):
        # Preserve the real tokenizer vocabulary. This catches model/tokenizer
        # compatibility errors during the smoke test.
        config.vocab_size = base_config.vocab_size
    if hasattr(config, "layer_types") and hasattr(config, "num_hidden_layers"):
        # Transformers validates Olmo3's per-layer attention metadata when the
        # config is saved. Keep the metadata length aligned with the shrunken
        # layer count.
        layer_types = list(getattr(base_config, "layer_types", []))
        if layer_types:
            config.layer_types = layer_types[: config.num_hidden_layers]
    return config


def load_tokenizer(model_name: str, model_cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def install_assistant_template(tokenizer, template_path: Path) -> None:
    """Install a template with generation markers for assistant-only loss."""
    tokenizer.chat_template = template_path.read_text()


def load_model(model_name: str, model_cfg: dict[str, Any]):
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    if model_cfg.get("smoke", False):
        base_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        smoke_config = build_smoke_config(
            base_config, model_cfg.get("smoke_overrides", {})
        )
        model = AutoModelForCausalLM.from_config(
            smoke_config, trust_remote_code=trust_remote_code
        )
    else:
        dtype = torch.bfloat16 if model_cfg.get("bf16", True) else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    return model


def synthetic_dataset(size: int) -> Dataset:
    rows = []
    for index in range(size):
        rows.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"What is the result of {index} plus 1?",
                    },
                    {"role": "assistant", "content": f"The result is {index + 1}."},
                ]
            }
        )
    return Dataset.from_list(rows)


def normalize_conversation_columns(dataset: Dataset) -> Dataset:
    columns = set(dataset.column_names)
    if "messages" in columns or "text" in columns:
        return dataset
    if "conversations" in columns:
        return dataset.rename_column("conversations", "messages")
    raise ValueError(
        f"Expected a messages, conversations, or text column; found {sorted(columns)}"
    )


def load_train_eval_data(data_cfg: dict[str, Any], seed: int) -> tuple[Dataset, Dataset]:
    if data_cfg.get("synthetic", False):
        full = synthetic_dataset(data_cfg.get("train_size", 32) + data_cfg.get("eval_size", 8))
        split = full.train_test_split(
            test_size=data_cfg.get("eval_size", 8), seed=seed
        )
        return split["train"], split["test"]

    dataset_name = data_cfg["dataset_name"]
    train_split = data_cfg.get("train_split", "train")
    dataset = load_dataset(dataset_name, split=train_split)
    dataset = normalize_conversation_columns(dataset)
    sample_size = data_cfg.get("sample_size")
    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    split = dataset.train_test_split(
        test_size=data_cfg.get("eval_size", 2000), seed=seed
    )
    return split["train"], split["test"]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_loader(
    model,
    dataloader: DataLoader,
    label_smoothing: float,
    base_model=None,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    if base_model is not None:
        base_model.to(device)
        base_model.eval()

    totals = {
        "hard_nll": 0.0,
        "smoothed_nll": 0.0,
        "accuracy": 0.0,
        "entropy": 0.0,
        "kl_sft_base": 0.0,
        "tokens": 0,
    }

    for batch in dataloader:
        batch = move_batch(batch, device)
        labels = batch["labels"]
        outputs = model(**batch)
        logits = outputs.logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        mask = shifted_labels.ne(-100)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = shifted_labels.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = shifted_labels.clamp_min(0)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        nll = -token_log_probs[mask]
        smooth_loss = -log_probs.mean(dim=-1)[mask]

        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)[mask]
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(shifted_labels)[mask]

        count = int(mask.sum().item())
        totals["hard_nll"] += float(nll.sum().item())
        totals["smoothed_nll"] += float(
            ((1.0 - label_smoothing) * nll + label_smoothing * smooth_loss).sum().item()
        )
        totals["accuracy"] += float(correct.sum().item())
        totals["entropy"] += float(entropy.sum().item())
        totals["tokens"] += count

        if base_model is not None:
            base_outputs = base_model(**batch)
            base_log_probs = F.log_softmax(base_outputs.logits[:, :-1, :].float(), dim=-1)
            kl = (probabilities * (log_probs - base_log_probs)).sum(dim=-1)[mask]
            totals["kl_sft_base"] += float(kl.sum().item())

    tokens = max(totals.pop("tokens"), 1)
    return {key: value / tokens for key, value in totals.items()}


class CheckpointMetricsCallback(TrainerCallback):
    def __init__(self, label_smoothing: float, measure_kl: bool):
        self.label_smoothing = label_smoothing
        self.measure_kl = measure_kl
        self.trainer = None
        self.records: list[dict[str, Any]] = []

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.trainer is None:
            return control
        model = model or self.trainer.model
        base_model = self.trainer.base_model_for_metrics if self.measure_kl else None
        metrics = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train": evaluate_loader(
                model,
                self.trainer.get_train_dataloader(),
                self.label_smoothing,
            ),
            "eval": evaluate_loader(
                model,
                self.trainer.get_eval_dataloader(),
                self.label_smoothing,
                base_model=base_model,
            ),
        }
        output_path = Path(args.output_dir) / f"metrics_epoch_{state.epoch:g}.json"
        output_path.write_text(json.dumps(metrics, indent=2) + "\n")
        self.records.append(metrics)
        model.train()
        return control


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    metrics_cfg = config.get("metrics", {})
    label_smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else float(train_cfg.get("label_smoothing", 0.0))
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        base_output_dir = Path(train_cfg["output_dir"])
        output_dir = base_output_dir.parent / (
            f"{base_output_dir.name}-{label_smoothing_tag(label_smoothing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "run_config.yaml")

    model_name = model_cfg["name_or_path"]
    tokenizer = load_tokenizer(model_name, model_cfg)
    if train_cfg.get("assistant_only_loss", False):
        template_path = Path(__file__).with_name("olmo3_assistant_template.jinja")
        install_assistant_template(tokenizer, template_path)

    model = load_model(model_name, model_cfg)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads") and train_cfg.get(
        "gradient_checkpointing", False
    ):
        model.enable_input_require_grads()

    base_model = None
    measure_kl = metrics_cfg.get("measure_kl", True) and not args.no_kl
    if measure_kl:
        base_model = copy.deepcopy(model)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)

    train_dataset, eval_dataset = load_train_eval_data(data_cfg, seed)
    callback = CheckpointMetricsCallback(label_smoothing, measure_kl)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 8)),
        learning_rate=3e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        label_smoothing_factor=label_smoothing,
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        logging_strategy="steps",
        logging_steps=float(train_cfg.get("logging_steps", 1)),
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(train_cfg.get("save_total_limit", 8)),
        # Keep model weights, optimizer, scheduler, RNG, and Trainer state in
        # each checkpoint so it remains usable for inspection or resume.
        save_only_model=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataset_text_field="text",
        max_length=int(train_cfg.get("max_length", 256)),
        packing=False,
        assistant_only_loss=bool(train_cfg.get("assistant_only_loss", False)),
        # TRL 1.13 defaults to chunked_nll, which requires a patched model
        # output containing num_valid_tokens. Standard causal-LM NLL is the
        # portable path for both the smoke model and Olmo 3.
        loss_type="nll",
        dataset_num_proc=None,
        remove_unused_columns=False,
        use_cache=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.base_model_for_metrics = base_model
    callback.trainer = trainer

    print(
        json.dumps(
            {
                "model": model_name,
                "smoke": bool(model_cfg.get("smoke", False)),
                "parameters": model.num_parameters(),
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "label_smoothing": label_smoothing,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    trainer.train()
    trainer.save_model(output_dir / "final")


if __name__ == "__main__":
    main()
