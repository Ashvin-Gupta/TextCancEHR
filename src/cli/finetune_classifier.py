"""
CLI: Fine-tune an LLM-backed binary classifier on EHR narratives.

This is the "real" entrypoint. `src/pipelines/finetune_llm_classifier.py` remains
for backwards compatibility with old scripts.
"""

from __future__ import annotations

import argparse
import os
import pprint

import torch
import wandb
import yaml
from huggingface_hub import login

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier, run_classification_training
from src.training.utils import load_LoRA_model, compute_and_sort_by_length, seed_all
from src.tokenization.ehr_special_tokens import EHRTokenExtensionStaticTokenizer


EXPERIMENT_NO_PRETRAIN = "no_pretrain"
EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER = "pretrained_cls"
EXPERIMENT_PRETRAIN_CLASSIFIER_LORA = "pretrained_cls_lora"


def load_model_for_mode(config: dict, experiment_mode: str):
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]

    if experiment_mode == EXPERIMENT_NO_PRETRAIN:
        translator = EHRTokenExtensionStaticTokenizer()
        model, tokenizer = translator.extend_tokenizer(
            model_name=model_config["unsloth_model"],
            max_seq_length=data_config["max_length"],
            load_in_4bit=training_config.get("load_in_4bit", True),
        )
        print("\nLoaded base model without continued pretraining. Only the classifier head will train.")
        return model, tokenizer

    if experiment_mode in (EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER, EXPERIMENT_PRETRAIN_CLASSIFIER_LORA):
        if not model_config.get("pretrained_checkpoint"):
            raise ValueError(f"'model.pretrained_checkpoint' must be set for experiment mode '{experiment_mode}'.")
        return load_LoRA_model(config)

    raise ValueError(f"Unknown experiment mode '{experiment_mode}'.")


def _load_hf_token_from_resources() -> str | None:
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if not os.path.exists(token_file):
        return None
    try:
        with open(token_file, "r") as f:
            return f.readline().split("=")[1].strip('"').strip() or None
    except Exception:
        return None


def main(config_path: str):
    seed_all(42)

    print("=" * 80)
    print("LLM Binary Classification Fine-tuning")
    print("=" * 80)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loaded configuration")
    pprint.pprint(config)
    print("=" * 80)

    model_config = config["model"]
    experiment_config = config.get("experiment", {})
    experiment_mode = experiment_config.get("mode", EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER)
    data_config = config["data"]
    training_config = config["training"]
    wandb_config = config.get("wandb", {})

    mode_msg = {
        EXPERIMENT_NO_PRETRAIN: "No continued pretraining - classifier head only.",
        EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER: "Using continued-pretrained checkpoint - classifier head only.",
        EXPERIMENT_PRETRAIN_CLASSIFIER_LORA: "Using continued-pretrained checkpoint - training classifier head + LoRA adapters.",
    }[experiment_mode]
    print(f"Experiment mode: {experiment_mode} -> {mode_msg}")

    if wandb_config.get("enabled", False):
        run_name = wandb_config.get("run_name") or f"classifier_{config.get('name', 'default')}"
        wandb.init(project=wandb_config.get("project", "llm-classification"), name=run_name, config=config)
        print(f"\nWandB enabled - Project: {wandb_config.get('project')}, Run: {run_name}")

    hf_token = _load_hf_token_from_resources()
    require_login = experiment_config.get("force_hf_login", False) or experiment_mode != EXPERIMENT_NO_PRETRAIN
    if hf_token and require_login:
        try:
            login(token=str(hf_token))
            print("HuggingFace login successful.")
        except Exception as e:
            print(f"Failed to login to HuggingFace: {e}")
    elif require_login:
        print("No HuggingFace token available but login required for this mode.")
    else:
        print("Skipping HuggingFace login for classifier-only experiment.")

    model, tokenizer = load_model_for_mode(config, experiment_mode)

    print("\n" + "=" * 80)
    print("Creating LLM Classifier wrapper...")
    print("=" * 80)

    multi_label_task = bool(training_config.get("multi_label", False))
    train_lora_adapters = bool(model_config.get("train_lora", False))
    if "freeze_lora" in model_config:
        train_lora_adapters = not bool(model_config["freeze_lora"])
    if experiment_mode == EXPERIMENT_PRETRAIN_CLASSIFIER_LORA:
        train_lora_adapters = True

    freeze_llm = bool(model_config.get("freeze_llm", True))
    trainable_keywords = ["lora_"] if train_lora_adapters else None

    classifier_model = LLMClassifier(
        base_model=model,
        hidden_size=model_config["hidden_size"],
        num_labels=model_config["num_labels"],
        freeze_base=freeze_llm,
        trainable_param_keywords=trainable_keywords,
        multi_label=multi_label_task,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)

    dataset_args = dict(
        data_dir=data_config["data_dir"],
        vocab_file=data_config["vocab_filepath"],
        labels_file=data_config["labels_filepath"],
        medical_lookup_file=data_config["medical_lookup_filepath"],
        lab_lookup_file=data_config["lab_lookup_filepath"],
        region_lookup_file=data_config["region_lookup_filepath"],
        time_lookup_file=data_config["time_lookup_filepath"],
        format="text",
        cutoff_months=data_config.get("cutoff_months", 1),
        max_sequence_length=None,
        tokenizer=None,
        data_type=data_config.get("data_type", "raw"),
    )

    train_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    test_dataset = UnifiedEHRDataset(split="held_out", **dataset_args)

    if data_config.get("sort_by_length", True):
        print("\n" + "=" * 80)
        print("Sorting datasets by sequence length for efficient batching...")
        print("=" * 80)
        train_dataset = compute_and_sort_by_length(train_dataset, tokenizer, shuffle_buckets=True, num_buckets=20)
        val_dataset = compute_and_sort_by_length(val_dataset, tokenizer, shuffle_buckets=False)
        test_dataset = compute_and_sort_by_length(test_dataset, tokenizer, shuffle_buckets=False)
        print("  ✓ Datasets sorted by length")

    print(f"  - Train dataset: {len(train_dataset)} patients")
    print(f"  - Validation dataset: {len(val_dataset)} patients")
    print(f"  - Test dataset: {len(test_dataset)} patients")

    print("\n" + "=" * 80)
    print("Creating data collator...")
    print("=" * 80)

    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config.get("max_length"),
        binary_classification=not multi_label_task,
        truncation=False,
        handle_long_sequences=data_config.get("handle_long_sequences", "warn"),
    )

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    trainer, eval_results = run_classification_training(
        config=config,
        model=classifier_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        collate_fn=collate_fn,
    )
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal model saved to: {training_config['output_dir']}/final_model")
    return trainer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Binary Classification Fine-tuning")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file")
    args = parser.parse_args()
    main(args.config_filepath)

