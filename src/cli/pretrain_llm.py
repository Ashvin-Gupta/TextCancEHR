"""
CLI: Continued pretraining of a language model on EHR narratives.

This is the "real" entrypoint. `src/pipelines/llm_pretrain2.py` is kept as a
thin wrapper for backwards compatibility with old job scripts.
"""

from __future__ import annotations

import argparse
import os
import pprint
from datetime import datetime
from threading import Thread
import textwrap

from unsloth import FastLanguageModel
import torch
import wandb
import yaml
from huggingface_hub import login
from transformers import TrainerCallback, TrainingArguments, TextIteratorStreamer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.data.unified_dataset import UnifiedEHRDataset
from src.tokenization.ehr_special_tokens import EHRTokenExtensionStaticTokenizer
from src.training.utils import seed_all


class InferenceCallback(TrainerCallback):
    """Runs a quick generation sanity-check at the end of each epoch."""

    def __init__(self, model, tokenizer, prompt: str):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n" + "=" * 80)
        print(f"Running inference test after epoch {state.epoch}...")
        print("=" * 80)
        run_ehr_inference(self.model, self.tokenizer, self.prompt)
        print("\n")


DEFAULT_INFERENCE_PROMPT = (
    "<start> <DEMOGRAPHIC> AGE: 70-74 <DEMOGRAPHIC> GENDER FEMALE <DEMOGRAPHIC> "
    "ETHNICITY WHITE <DEMOGRAPHIC> REGION South West <EVENT> Myocardial Infarction "
    "<TIME> 4mt-6mt <EVENT> Chronic Kidney Disease <TIME> 4d-7d <unknown> "
    "<TIME> 24mt-60mt <EVENT> Hypertension <EVENT> Transient Ischaemic Attack "
    "<EVENT> Cholesterolaemia <TIME> 8mt-10mt <EVENT> Death of husband "
    "<TIME> 24mt-60mt <EVENT> Liver abscess - excluding amoebic liver abscess "
    "<EVENT> Gallbladder Disease <TIME> 2d-4d <EVENT> Hernia Diaphragm "
    "<TIME> 2mt-4mt <EVENT> Clouded consciousness <TIME> 2mt-4mt <EVENT> "
    "Noninfectious enteritis <TIME> 12d-20d <EVENT> Syncope and collapse <EVENT>"
)


def extract_text(base_dataset, tokenizer):
    """Extract all valid text narratives from the dataset."""
    text_list = []

    print(f"  - Processing {len(base_dataset)} patients...")
    for i in range(len(base_dataset)):
        item = base_dataset[i]
        if item is not None:
            text = item["text"]
            text = text.replace("<start>", "").replace("<end>", "").strip()
            text_list.append(text)

    print(f"  - Extracted {len(text_list)} valid narratives.")
    return text_list


def verify_patient(train_text_list, tokenizer):
    print("\nVerifying data - First 1 patient narratives:")
    for i in range(min(1, len(train_text_list))):
        print(f"\n--- PATIENT {i} ---")
        print(f"{train_text_list[i][:1000]}...")

        print(f"\n--- PATIENT {i} TOKENIZATION ---")
        tokens = tokenizer.tokenize(train_text_list[i])
        token_ids = tokenizer.encode(train_text_list[i], add_special_tokens=False)

        print(f"Text length: {len(train_text_list[i])} characters")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Number of token IDs: {len(token_ids)}")
        print(f"First 100 tokens: {tokens[:100]}")
        print(f"First 100 token IDs: {token_ids[:100]}")

        print("Token-to-text mapping (first 10):")
        for j in range(min(100, len(tokens))):
            decoded = tokenizer.decode([token_ids[j]])
            print(f"  Token {j}: '{tokens[j]}' -> ID {token_ids[j]} -> Decoded: '{decoded}'")

    print("\n" + "=" * 80)
    print("Creating SFT datasets...")
    print("=" * 80)


def run_ehr_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256, max_print_width: int = 100):
    """Stream a short greedy-ish generation for sanity-checking."""
    model.eval()
    FastLanguageModel.for_inference(model)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    print("-" * 80)
    print(f"PROMPT:\n{textwrap.fill(prompt, width=max_print_width)}\n")
    print("MODEL GENERATION:")
    print("-" * 80)

    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = dict(
        inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=30,
        repetition_penalty=1.2,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("MODEL OUTPUT (RAW STREAM):")
    for new_text in text_streamer:
        print(new_text, end="")
    print("\n")


def _load_hf_token_from_resources() -> str | None:
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if not os.path.exists(token_file):
        return None
    try:
        with open(token_file, "r") as f:
            line = f.readline().strip()
        return line.split("=")[1].strip('"').strip() or None
    except Exception:
        return None


def main(config_path: str):
    seed_all(42)

    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loaded configuration")
    pprint.pprint(config)
    print("=" * 80)

    wandb_config = config.get("wandb", {})
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    lora_config = config["lora"]

    model_name = model_config["model_name"].split("/")[-1]
    default_run_name = (
        f"{model_name}-pretrain-{data_config['cutoff_months']}month-cutoff"
        f"_r{lora_config['r']}"
        f"_alpha{lora_config['lora_alpha']}"
        f"_dropout{lora_config['lora_dropout']}"
        f"_lr{training_config['learning_rate']}"
        f"_bs{training_config['batch_size']}"
        f"_wd{training_config['weight_decay']}"
        f"_ga{training_config['gradient_accumulation_steps']}"
        f"_length{model_config['max_length']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if wandb_config.get("enabled", False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "ehr-llm-pretraining")
        run_name = wandb_config.get("run_name", default_run_name)
        report_to = "wandb"
        if local_rank == 0:
            wandb.init(project=os.environ["WANDB_PROJECT"], config=config, name=run_name)
            wandb.config.update(config, allow_val_change=True)
    else:
        run_name = default_run_name
        report_to = "none"

    hf_token = _load_hf_token_from_resources()
    if hf_token:
        login(token=str(hf_token))
    else:
        print("No HuggingFace token provided - skipping login.")

    print("\n" + "=" * 80)
    print(f"Loading model: {model_config['model_name']}")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    else:
        print("  - CUDA not available, using CPU")

    translator = EHRTokenExtensionStaticTokenizer()
    model, tokenizer = translator.extend_tokenizer(
        model_name=model_config["model_name"],
        max_seq_length=model_config["max_length"],
        load_in_4bit=training_config.get("load_in_4bit", True),
    )

    print("\n" + "=" * 80)
    print("Creating datasets in 'text' format...")
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
        data_type=training_config.get("input_data", "binned"),
    )

    print("\nLoading training data...")
    train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    print(f"  - Loaded {len(train_base_dataset)} training patients")

    print("\nLoading validation data...")
    val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  - Loaded {len(val_base_dataset)} validation patients")

    train_text_list = extract_text(train_base_dataset, tokenizer)
    val_text_list = extract_text(val_base_dataset, tokenizer)
    verify_patient(train_text_list, tokenizer)

    train_dataset = Dataset.from_dict({"text": train_text_list})
    val_dataset = Dataset.from_dict({"text": val_text_list})

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("r", 16),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        ),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing=training_config.get("gradient_checkpointing", "unsloth"),
        random_state=42,
        use_rslora=lora_config.get("use_rslora", True),
        loftq_config=lora_config.get("loftq_config", None),
    )
    print("  - Applied LoRA adapters (PEFT) to the model.")

    training_args = SFTConfig(
        dataloader_num_workers=training_config.get("dataloader_num_workers", 8),
        output_dir=training_config["output_dir"],
        overwrite_output_dir=training_config.get("overwrite_output_dir", True),
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config.get("eval_batch_size", training_config["batch_size"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        warmup_steps=training_config.get("warmup_steps", 500),
        logging_steps=training_config.get("logging_steps", 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 2),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        report_to=report_to,
        run_name=run_name,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataset_text_field="text",
        max_seq_length=model_config["max_length"],
        packing=True,
    )

    inference_callback = InferenceCallback(model, tokenizer, DEFAULT_INFERENCE_PROMPT)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[inference_callback],
    )

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    trainer.train()

    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    final_subdir = training_config.get("final_subdir", "final_model")
    final_model_path = os.path.join(training_config["output_dir"], final_subdir)
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"  - Model + tokenizer saved to: {final_model_path}")

    if val_dataset:
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        eval_results = trainer.evaluate()
        print("\nValidation Results:")
        for key, value in eval_results.items():
            try:
                print(f"  - {key}: {value:.4f}")
            except Exception:
                print(f"  - {key}: {value}")

    print("\n" + "=" * 80)
    print("Running final inference test...")
    print("=" * 80)
    run_ehr_inference(model, tokenizer, DEFAULT_INFERENCE_PROMPT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Continued Pretraining on EHR Data")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file")
    args = parser.parse_args()
    main(args.config_filepath)

