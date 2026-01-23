# src/pipelines/text_based/finetune_motor.py

"""
MOTOR Time-to-Event Fine-tuning Script.

Fine-tunes a pretrained LLM with a MOTOR piecewise exponential head
for survival analysis / time-to-event prediction.

This removes the dependency on a fixed time cutoff and instead learns
to predict risk over time using piecewise exponential hazards.

Usage:
    python -m src.pipelines.text_based.finetune_motor --config_filepath configs/motor_finetune.yaml
"""

import argparse
import yaml
import os
import wandb
import torch
import torch.nn as nn
import pprint
from huggingface_hub import login
import random
import pandas as pd

from unsloth import FastLanguageModel

from src.data.unified_dataset import UnifiedEHRDataset
from src.training.motor_trainer import (
    LLMMotorModel,
    run_motor_training,
)
from src.training.utils import load_LoRA_model, compute_and_sort_by_length, seed_all
from src.pipelines.text_based.token_adaption2 import EHRTokenExtensionStaticTokenizer

# Experiment modes (same as classifier)
EXPERIMENT_NO_PRETRAIN = "no_pretrain"
EXPERIMENT_PRETRAIN_ONLY_MOTOR = "pretrained_motor"
EXPERIMENT_PRETRAIN_MOTOR_LORA = "pretrained_motor_lora"


def load_model_for_motor(config: dict, experiment_mode: str):
    """
    Load the correct model/tokenizer pair based on the experiment mode.
    Same logic as classifier to ensure compatibility.
    
    Args:
        config: Configuration dictionary
        experiment_mode: One of EXPERIMENT_NO_PRETRAIN, EXPERIMENT_PRETRAIN_ONLY_MOTOR, 
                        or EXPERIMENT_PRETRAIN_MOTOR_LORA
    
    Returns:
        model, tokenizer: The loaded model and tokenizer
    """
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    
    if experiment_mode == EXPERIMENT_NO_PRETRAIN:
        # Load base model with token extension (no pretrained checkpoint)
        translator = EHRTokenExtensionStaticTokenizer()
        model, tokenizer = translator.extend_tokenizer(
            model_name=model_config['unsloth_model'],
            max_seq_length=data_config['max_length'],
            load_in_4bit=training_config.get('load_in_4bit', True)
        )
        print("\nLoaded base model without continued pretraining. Only the MOTOR head will train.")
        return model, tokenizer
    
    if experiment_mode in (EXPERIMENT_PRETRAIN_ONLY_MOTOR, EXPERIMENT_PRETRAIN_MOTOR_LORA):
        # Load pretrained model with LoRA adapters
        if not model_config.get('pretrained_checkpoint'):
            raise ValueError(
                f"'model.pretrained_checkpoint' must be set for experiment mode '{experiment_mode}'."
            )
        return load_LoRA_model(config)
    
    raise ValueError(f"Unknown experiment mode '{experiment_mode}'.")


class SurvivalCollator:
    """
    Data collator for MOTOR survival analysis.
    
    Handles tokenization and batching of text data along with
    event_times and event_indicators for survival analysis.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 8192,
        truncation: bool = True,
        handle_long_sequences: str = 'truncate'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.handle_long_sequences = handle_long_sequences
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dicts with 'text', 'time_to_event', 'event_indicator'
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'event_times', 'event_indicators'
        """
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        # Extract texts and survival data
        texts = [b['text'] for b in batch]
        event_times = torch.tensor(
            [b.get('time_to_event', 0.0) for b in batch],
            dtype=torch.float32
        )
        event_indicators = torch.tensor(
            [b.get('event_indicator', 0) for b in batch],
            dtype=torch.long
        )
        
        # Tokenize texts
        # Account for EOS token we'll add
        max_allowed = self.max_length - 1 if self.tokenizer.eos_token_id is not None else self.max_length
        
        # Process each text individually to handle long sequences
        all_input_ids = []
        all_attention_masks = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            encoded = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Handle long sequences
            if input_ids.size(1) > max_allowed:
                if self.handle_long_sequences == 'truncate':
                    # Keep most recent (rightmost) tokens
                    input_ids = input_ids[:, -max_allowed:]
                    attention_mask = attention_mask[:, -max_allowed:]
                elif self.handle_long_sequences == 'skip':
                    continue
                elif self.handle_long_sequences == 'error':
                    raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_length {max_allowed}")
                else:  # 'warn' or default
                    print(f"Warning: Truncating sequence from {input_ids.size(1)} to {max_allowed}")
                    input_ids = input_ids[:, -max_allowed:]
                    attention_mask = attention_mask[:, -max_allowed:]
            
            # Append EOS token
            if self.tokenizer.eos_token_id is not None:
                eos_tensor = torch.tensor([[self.tokenizer.eos_token_id]], dtype=input_ids.dtype)
                eos_mask = torch.ones((1, 1), dtype=attention_mask.dtype)
                input_ids = torch.cat([input_ids, eos_tensor], dim=1)
                attention_mask = torch.cat([attention_mask, eos_mask], dim=1)
            
            all_input_ids.append(input_ids.squeeze(0))
            all_attention_masks.append(attention_mask.squeeze(0))
            valid_indices.append(i)
        
        if len(all_input_ids) == 0:
            return None
        
        # Pad to max length in batch
        max_len_in_batch = max(ids.size(0) for ids in all_input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len_in_batch - input_ids.size(0)
            if pad_len > 0:
                # Left padding for causal LM
                pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                pad_mask = torch.zeros(pad_len, dtype=attention_mask.dtype)
                input_ids = torch.cat([pad_ids, input_ids])
                attention_mask = torch.cat([pad_mask, attention_mask])
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
        
        # Stack into tensors
        input_ids_tensor = torch.stack(padded_input_ids)
        attention_mask_tensor = torch.stack(padded_attention_masks)
        
        # Filter survival data to match valid indices
        event_times = event_times[valid_indices]
        event_indicators = event_indicators[valid_indices]
        
        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'event_times': event_times,
            'event_indicators': event_indicators,
        }


class SurvivalDatasetWrapper:
    """
    Wrapper that adds survival data to UnifiedEHRDataset.
    
    Computes time_to_event and event_indicator from existing labels and dates.
    """
    
    def __init__(self, base_dataset: UnifiedEHRDataset, reference_date_method: str = 'first_event'):
        """
        Args:
            base_dataset: UnifiedEHRDataset instance
            reference_date_method: How to compute reference date
                - 'first_event': Use first timestamp in patient record
                - 'fixed': Use a fixed date (e.g., start of study)
        """
        self.base_dataset = base_dataset
        self.reference_date_method = reference_date_method
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base sample
        sample = self.base_dataset[idx]
        if sample is None:
            return None
        
        # Get patient record for timestamp access
        patient_record = self.base_dataset.patient_records[idx]
        subject_id = patient_record['subject_id']
        timestamps = patient_record['timestamps']
        
        # Get label and cancer date
        label = self.base_dataset.subject_to_label.get(subject_id, 0)
        cancer_date = self.base_dataset.subject_to_cancer_date.get(subject_id)
        
        # Compute reference date (start of observation)
        valid_timestamps = [ts for ts in timestamps if ts > 0]
        if valid_timestamps:
            first_timestamp = min(valid_timestamps)
            last_timestamp = max(valid_timestamps)
            reference_date = pd.Timestamp.fromtimestamp(first_timestamp)
            last_date = pd.Timestamp.fromtimestamp(last_timestamp)
        else:
            # Fallback: use cancer date - 5 years or current date
            if pd.notna(cancer_date):
                reference_date = cancer_date - pd.DateOffset(years=5)
                last_date = cancer_date
            else:
                reference_date = pd.Timestamp.now() - pd.DateOffset(years=5)
                last_date = pd.Timestamp.now()
        
        # Compute time_to_event and event_indicator
        if label > 0 and pd.notna(cancer_date):
            # Case: cancer occurred
            time_to_event = (cancer_date - reference_date).days
            event_indicator = 1
        else:
            # Control: censored at last observation
            time_to_event = (last_date - reference_date).days
            event_indicator = 0
        
        # Ensure non-negative time
        time_to_event = max(0, time_to_event)
        
        # Add survival data to sample
        sample['time_to_event'] = time_to_event
        sample['event_indicator'] = event_indicator
        
        return sample


def main(config_path: str):
    """Main function for MOTOR time-to-event fine-tuning."""
    
    # Set seed
    seed_all(42)
    
    print("=" * 80)
    print("MOTOR Time-to-Event Fine-tuning")
    print("=" * 80)
    
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print('Loaded configuration')
    pprint.pprint(config)
    print("=" * 80)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    motor_config = config.get('motor', {})
    wandb_config = config.get('wandb', {})
    
    # 2. Set up WandB
    if wandb_config.get('enabled', False):
        run_name = wandb_config.get("run_name", "motor-tte")
        wandb.init(
            project=wandb_config.get("project", "motor-tte"),
            name=run_name,
            config=config
        )
        print(f"\nWandB enabled - Project: {wandb_config['project']}, Run: {run_name}")
    
    # 3. HuggingFace Login
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    hf_token = None
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                hf_token = f.readline().split('=')[1].strip('"')
        except Exception as e:
            print(f"Failed to read HF token: {e}")
    
    if hf_token:
        try:
            login(token=str(hf_token))
            print("HuggingFace login successful.")
        except Exception as e:
            print(f"Failed to login to HuggingFace: {e}")
    
    # 4. Load Model and Tokenizer (same logic as classifier)
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    
    experiment_mode = config.get('experiment', {}).get('mode', EXPERIMENT_PRETRAIN_ONLY_MOTOR)
    
    mode_msg = {
        EXPERIMENT_NO_PRETRAIN: "No continued pretraining - MOTOR head only.",
        EXPERIMENT_PRETRAIN_ONLY_MOTOR: "Using continued-pretrained checkpoint - MOTOR head only.",
        EXPERIMENT_PRETRAIN_MOTOR_LORA: "Using continued-pretrained checkpoint - training MOTOR head + LoRA adapters."
    }.get(experiment_mode, f"Unknown mode: {experiment_mode}")
    print(f"Experiment mode: {experiment_mode} -> {mode_msg}")
    
    model, tokenizer = load_model_for_motor(config, experiment_mode)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # 5. Wrap model with MOTOR head
    print("\n" + "=" * 80)
    print("Creating MOTOR model wrapper...")
    print("=" * 80)
    
    # Parse piece boundaries if provided
    piece_boundaries = motor_config.get('piece_boundaries', None)
    if piece_boundaries:
        # Convert from list of lists to list of tuples
        piece_boundaries = [tuple(b) for b in piece_boundaries]
    
    # Determine if we should train LoRA adapters based on experiment mode
    train_lora_adapters = False
    if experiment_mode == EXPERIMENT_PRETRAIN_MOTOR_LORA:
        train_lora_adapters = True
    elif model_config.get('train_lora', False):
        train_lora_adapters = True
    
    trainable_keywords = ["lora_"] if train_lora_adapters else None
    freeze_llm = model_config.get('freeze_llm', True)
    
    print(f"  - Freeze LLM: {freeze_llm}")
    print(f"  - Train LoRA adapters: {train_lora_adapters}")
    
    motor_model = LLMMotorModel(
        base_model=model,
        hidden_size=model_config['hidden_size'],
        num_pieces=motor_config.get('num_pieces', 6),
        piece_boundaries=piece_boundaries,
        intermediate_dim=motor_config.get('intermediate_dim', 64),
        freeze_base=freeze_llm,
        trainable_param_keywords=trainable_keywords,
        tokenizer=tokenizer
    )
    
    # 6. Load Datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "time_lookup_file": data_config["time_lookup_filepath"],
        "format": 'text',
        "cutoff_months": None,  # No cutoff for MOTOR!
        "max_sequence_length": None,
        "data_type": data_config.get('data_type', 'raw')
    }
    
    # Load base datasets
    train_base = UnifiedEHRDataset(split="train", **dataset_args)
    val_base = UnifiedEHRDataset(split="tuning", **dataset_args)
    test_base = UnifiedEHRDataset(split="held_out", **dataset_args)
    
    # Wrap with survival data
    train_dataset = SurvivalDatasetWrapper(train_base)
    val_dataset = SurvivalDatasetWrapper(val_base)
    test_dataset = SurvivalDatasetWrapper(test_base)
    
    print(f"  - Train dataset: {len(train_dataset)} patients")
    print(f"  - Validation dataset: {len(val_dataset)} patients")
    print(f"  - Test dataset: {len(test_dataset)} patients")
    
    # 7. Sample data verification
    print("\n" + "=" * 80)
    print("Sample survival data verification:")
    print("=" * 80)
    
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        if sample is not None:
            print(f"\nPatient {i}:")
            print(f"  - Event indicator: {sample['event_indicator']} ({'Event' if sample['event_indicator'] == 1 else 'Censored'})")
            print(f"  - Time to event: {sample['time_to_event']:.0f} days ({sample['time_to_event']/365:.1f} years)")
            print(f"  - Text length: {len(sample['text'])} chars")
            print(f"  - Text preview: {sample['text'][:200]}...")
    
    # 8. Create Data Collator
    print("\n" + "=" * 80)
    print("Creating survival data collator...")
    print("=" * 80)
    
    collate_fn = SurvivalCollator(
        tokenizer=tokenizer,
        max_length=data_config.get('max_length', 8192),
        truncation=True,
        handle_long_sequences=data_config.get('handle_long_sequences', 'truncate')
    )
    
    # 9. Train
    print("\n" + "=" * 80)
    print("Starting MOTOR training...")
    print("=" * 80)
    
    trainer, eval_results = run_motor_training(
        config=config,
        model=motor_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        collate_fn=collate_fn
    )
    
    print("\n" + "=" * 80)
    print("MOTOR Training Complete!")
    print("=" * 80)
    print(f"\nFinal model saved to: {training_config['output_dir']}/final_model")
    
    return trainer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOTOR Time-to-Event Fine-tuning")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    main(args.config_filepath)
