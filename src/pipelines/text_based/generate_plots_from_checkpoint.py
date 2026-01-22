#!/usr/bin/env python3
"""
Generate ROC and Precision-Recall curves from a saved checkpoint.
"""

import argparse
import yaml
import os
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import load_file
import glob

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier
from src.training.utils import load_LoRA_model
from src.evaluations.visualisation import plot_classification_performance


def load_checkpoint_state_dict(checkpoint_path):
    """
    Load state dict from checkpoint, handling different save formats.
    """
    # Check what files exist in the checkpoint
    print(f"Checking checkpoint directory: {checkpoint_path}")
    
    # Option 1: model.safetensors (new HF format)
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        print(f"Found model.safetensors, loading...")
        return load_file(safetensors_path)
    
    # Option 2: pytorch_model.bin (old format)
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        print(f"Found pytorch_model.bin, loading...")
        return torch.load(pytorch_path, map_location='cpu')
    
    # Option 3: Sharded checkpoints (pytorch_model-*.bin)
    sharded_files = glob.glob(os.path.join(checkpoint_path, "pytorch_model-*.bin"))
    if sharded_files:
        print(f"Found {len(sharded_files)} sharded checkpoint files, loading...")
        state_dict = {}
        for shard_file in sorted(sharded_files):
            shard_dict = torch.load(shard_file, map_location='cpu')
            state_dict.update(shard_dict)
        return state_dict
    
    # Option 4: Sharded safetensors (model-*.safetensors)
    sharded_safetensors = glob.glob(os.path.join(checkpoint_path, "model-*.safetensors"))
    if sharded_safetensors:
        print(f"Found {len(sharded_safetensors)} sharded safetensors files, loading...")
        state_dict = {}
        for shard_file in sorted(sharded_safetensors):
            shard_dict = load_file(shard_file)
            state_dict.update(shard_dict)
        return state_dict
    
    # List what files are actually there
    files = os.listdir(checkpoint_path)
    raise FileNotFoundError(
        f"Could not find model weights in {checkpoint_path}.\n"
        f"Files found: {files}\n"
        f"Expected one of: model.safetensors, pytorch_model.bin, or sharded files"
    )


def generate_plots(config_path: str, checkpoint_path: str, dataset_split: str = "tuning"):
    """
    Generate ROC and PR curves from a checkpoint.
    
    Args:
        config_path: Path to the config YAML file
        checkpoint_path: Path to the saved model checkpoint
        dataset_split: Which dataset to evaluate ("tuning" for validation, "held_out" for test)
    """
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # First, try to load tokenizer from checkpoint
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("Loaded tokenizer from checkpoint")
    except:
        print("Could not load tokenizer from checkpoint, loading from base model...")
        # Load base model to get tokenizer
        base_model, tokenizer = load_LoRA_model(config)
    
    # Reconstruct the model architecture
    print("Reconstructing model architecture...")
    
    # Load base model with LoRA if we haven't already
    if 'base_model' not in locals():
        base_model, tokenizer = load_LoRA_model(config)
    
    # Wrap with classifier
    model = LLMClassifier(
        base_model=base_model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=True,
        trainable_param_keywords=["lora_"] if model_config.get('train_lora', False) else None,
        multi_label=training_config.get('multi_label', False),
        tokenizer=tokenizer
    )
    
    # Load the saved state dict
    print("Loading checkpoint weights...")
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    
    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"\nLoading {dataset_split} dataset...")
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "time_lookup_file": data_config["time_lookup_filepath"],
        "format": 'text',
        "cutoff_months": data_config.get("cutoff_months", 1),
        "max_sequence_length": None,
        "tokenizer": None,
        "data_type": data_config.get('data_type', 'raw')
    }
    
    dataset = UnifiedEHRDataset(split=dataset_split, **dataset_args)
    print(f"Loaded {len(dataset)} patients")
    
    # Create collator
    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config.get('max_length'),
        binary_classification=not training_config.get('multi_label', False),
        truncation=False,
        handle_long_sequences=data_config.get('handle_long_sequences', 'warn')
    )
    
    # Create a minimal trainer just for prediction
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        per_device_eval_batch_size=training_config.get('eval_batch_size', 2),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 8),
        remove_unused_columns=False,
        bf16=training_config.get('bf16', True),
        fp16=training_config.get('fp16', False),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
    )
    
    # Get predictions
    print("\nGenerating predictions...")
    pred_output = trainer.predict(dataset)
    
    # Convert logits to probabilities
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    multi_label = training_config.get('multi_label', False)
    if multi_label:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
    else:
        # Column 1 is probability of positive class (Cancer)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    
    labels = pred_output.label_ids
    
    # Generate plots
    plot_output_dir = os.path.join(training_config['output_dir'], f"plots_{dataset_split}")
    print(f"\nGenerating plots...")
    
    if multi_label:
        print("Multi-label not supported for plotting")
    else:
        plot_classification_performance(labels, probs, plot_output_dir)
        print(f"\n✓ Plots saved to: {plot_output_dir}")
        print(f"  - ROC curve: {os.path.join(plot_output_dir, 'roc_curve.png')}")
        print(f"  - PR curve: {os.path.join(plot_output_dir, 'pr_curve.png')}")
        print(f"  - Threshold analysis: {os.path.join(plot_output_dir, 'threshold_analysis.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="tuning",
        choices=["tuning", "held_out", "train"],
        help="Dataset split to evaluate (tuning=validation, held_out=test)"
    )
    
    args = parser.parse_args()
    generate_plots(args.config, args.checkpoint, args.split)