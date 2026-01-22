# src/pipelines/text_based/finetune_llm_classifier.py

"""
Main script for fine-tuning a pretrained LLM for binary classification.

Loads a pretrained LLM with LoRA adapters and extended tokenizer,
freezes the LLM, and trains only a classification head on top.
"""

import argparse
import yaml
import os
import wandb
import torch
import pprint
from huggingface_hub import login
import random

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier, run_classification_training
from src.training.utils import load_LoRA_model, compute_and_sort_by_length
from src.pipelines.text_based.token_adaption2 import EHRTokenExtensionStaticTokenizer
from src.training.utils import seed_all

EXPERIMENT_NO_PRETRAIN = "no_pretrain"
EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER = "pretrained_cls"
EXPERIMENT_PRETRAIN_CLASSIFIER_LORA = "pretrained_cls_lora"


def load_model_for_mode(config: dict, experiment_mode: str):
    """
    Load the correct model/tokenizer pair based on the experiment mode.
    """
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    
    if experiment_mode == EXPERIMENT_NO_PRETRAIN:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['unsloth_model'],
            max_seq_length=model_config['max_length'],
            dtype=None,
            load_in_4bit=training_config.get('load_in_4bit', True),
            # device_map={"": local_rank}
        )
        print("\nLoaded base model without continued pretraining. Only the classifier head will train.")
        return model, tokenizer
    
    if experiment_mode in (EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER, EXPERIMENT_PRETRAIN_CLASSIFIER_LORA):
        if not model_config.get('pretrained_checkpoint'):
            raise ValueError(
                f"'model.pretrained_checkpoint' must be set for experiment mode '{experiment_mode}'."
            )
        return load_LoRA_model(config)
    
    raise ValueError(f"Unknown experiment mode '{experiment_mode}'.")


def main(config_path: str):
    # Set seed
    seed_all(42)
    print("=" * 80)
    print("LLM Binary Classification Fine-tuning")
    print("=" * 80)
    
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print('Loaded configuration')
    pprint.pprint(config)
    print("=" * 80)
    
    model_config = config['model']
    experiment_config = config.get('experiment', {})
    experiment_mode = experiment_config.get('mode', EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER)
    data_config = config['data']
    training_config = config['training']
    wandb_config = config.get('wandb', {})
    
    mode_msg = {
        EXPERIMENT_NO_PRETRAIN: "No continued pretraining - classifier head only.",
        EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER: "Using continued-pretrained checkpoint - classifier head only.",
        EXPERIMENT_PRETRAIN_CLASSIFIER_LORA: "Using continued-pretrained checkpoint - training classifier head + LoRA adapters."
    }[experiment_mode]
    print(f"Experiment mode: {experiment_mode} -> {mode_msg}")
    
    # 2. Set up WandB
    if wandb_config.get('enabled', False):
        run_name = wandb_config.get("run_name")
        if run_name is None:
            # Auto-generate run name
            run_name = f"classifier_{config.get('name', 'default')}"
        
        # Initialize wandb run
        wandb.init(
            project=wandb_config.get("project", "llm-classification"),
            name=run_name,
            config=config
        )
        print(f"\nWandB enabled - Project: {wandb_config['project']}, Run: {run_name}")
    
    # 3. HuggingFace Login (skip for pure classifier-only mode unless forced)
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    hf_token = None
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                hf_token = f.readline().split('=')[1].strip('"')
        except Exception as e:
            print(f"Failed to read HF token: {e}")
    
    require_login = experiment_config.get('force_hf_login', False) or experiment_mode != EXPERIMENT_NO_PRETRAIN
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
    
    # 5. Wrap model with classification head
    print("\n" + "=" * 80)
    print("Creating LLM Classifier wrapper...")
    print("=" * 80)
    
    multi_label_task = bool(training_config.get('multi_label', False))
    if multi_label_task:
        print("Multi-label flag detected. Ensure datasets/collators emit multi-hot labels. Current metrics remain binary.")
    
    train_lora_adapters = bool(model_config.get('train_lora', False))
    if 'freeze_lora' in model_config:
        train_lora_adapters = not bool(model_config['freeze_lora'])
    if experiment_mode == EXPERIMENT_PRETRAIN_CLASSIFIER_LORA:
        train_lora_adapters = True
    
    freeze_llm = bool(model_config.get('freeze_llm', True))
    
    trainable_keywords = ["lora_"] if train_lora_adapters else None
    
    classifier_model = LLMClassifier(
        base_model=model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=freeze_llm,
        trainable_param_keywords=trainable_keywords,
        multi_label=multi_label_task,
        tokenizer=tokenizer # Needed to debug and decode last hidden state
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
        "format": 'text',  # Return text narratives
        "cutoff_months": data_config.get("cutoff_months", 1),
        "max_sequence_length": None,  # No truncation at dataset level
        "tokenizer": None,  # Not needed for 'text' format
        "data_type": data_config.get('data_type', 'raw')
    }
    
    train_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    test_dataset = UnifiedEHRDataset(split="held_out", **dataset_args)

    print("\n" + "=" * 80)
    print("Sample Patient Trajectories (for debugging)")
    print("=" * 80)

    # Get 3 random indices
    num_samples_to_show = 3
    valid_indices = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        if sample is not None:
            valid_indices.append(i)
        if len(valid_indices) >= 100:  # Sample from first 100 valid patients
            break

    if len(valid_indices) >= num_samples_to_show:
        random_indices = random.sample(valid_indices, num_samples_to_show)
    else:
        random_indices = valid_indices

    for idx_num, idx in enumerate(random_indices, 1):
        sample = train_dataset[idx]
        if sample is not None:
            text = sample['text']
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            
            # Tokenize to get length
            token_count = len(tokenizer.encode(text, add_special_tokens=True))
            
            print(f"\n{'─'*80}")
            print(f"Patient {idx_num} (Index {idx}):")
            print(f"  Label: {label} ({'Cancer' if label > 0 else 'Control'})")
            print(f"  Token count: {token_count}")
            print(f"  Text length: {len(text)} characters")
            print(f"{'─'*80}")
            
            # Print first 1000 characters
            print("First 1000 characters:")
            print(text[:1000])
            
            # Print last 1000 characters
            if len(text) > 1000:
                print(f"\n... [skipped {len(text) - 2000} characters] ...\n")
                print("Last 1000 characters:")
                print(text[-1000:])
            
            # Look for "Unknown" in the text
            unknown_count = text.count("Unknown")
            if unknown_count > 0:
                print(f"\n⚠️  WARNING: Found {unknown_count} 'Unknown' tokens in this trajectory!")
                
                # Show context around first few "Unknown" occurrences
                print("\nContext around 'Unknown' tokens (first 3 occurrences):")
                start = 0
                for occurrence_num in range(min(3, unknown_count)):
                    pos = text.find("Unknown", start)
                    if pos != -1:
                        context_start = max(0, pos - 100)
                        context_end = min(len(text), pos + 100)
                        context = text[context_start:context_end]
                        print(f"\n  Occurrence {occurrence_num + 1} (position {pos}):")
                        print(f"  ...{context}...")
                        start = pos + 1

    print("\n" + "=" * 80)
    print("End of sample trajectories")
    print("=" * 80 + "\n")

    use_length_sorting = data_config.get('sort_by_length', True)

    if use_length_sorting:
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
    print("Analyzing sequence lengths...")
    print("=" * 80)

    # Sample sequences to check lengths
    sample_size = min(500, len(train_dataset))
    sample_lengths = []

    print(f"Sampling {sample_size} training sequences...")
    for i in range(sample_size):
        sample = train_dataset[i]
        if sample is not None:
            # Tokenize to get actual length
            tokens = tokenizer.encode(sample['text'], add_special_tokens=True)
            sample_lengths.append(len(tokens))

    if sample_lengths:
        sample_lengths_sorted = sorted(sample_lengths)
        print(f"\nSequence Length Statistics (sample of {len(sample_lengths)} patients):")
        print(f"  - Min length: {min(sample_lengths)} tokens")
        print(f"  - Max length: {max(sample_lengths)} tokens")
        print(f"  - Mean length: {sum(sample_lengths) / len(sample_lengths):.1f} tokens")
        print(f"  - Median length: {sample_lengths_sorted[len(sample_lengths)//2]} tokens")
        print(f"  - 95th percentile: {sample_lengths_sorted[int(len(sample_lengths)*0.95)]} tokens")
        print(f"  - 99th percentile: {sample_lengths_sorted[int(len(sample_lengths)*0.99)]} tokens")
        
        model_max = data_config.get('max_length', 32768)
        num_exceeding = sum(1 for l in sample_lengths if l > model_max)
        print(f"\n  - Sequences exceeding model max ({model_max}): {num_exceeding} ({num_exceeding/len(sample_lengths)*100:.1f}%)")
        
        if num_exceeding > 0:
            print(f"\n  ⚠️  WARNING: Some sequences exceed the model's max_length!")
            print(f"     Consider setting 'handle_long_sequences' in your config to:")
            print(f"     - 'truncate': Keep most recent events (recommended for EHR)")
            print(f"     - 'warn': Truncate with warning (default)")
            print(f"     - 'skip': Skip these patients")
            print(f"     - 'error': Fail if any sequence is too long")
    else:
        print("  - No valid sequences found in sample")

    # Print a few examples
    print("\n" + "=" * 80)
    print("Sample data (last 500 chars):")
    print("=" * 80)
    for i in range(min(2, len(train_dataset))):
        sample = train_dataset[i]
        if sample is not None:
            text_preview = sample['text'][-500:] if len(sample['text']) > 500 else sample['text']
            label_tensor = sample['label']
            scalar_label = None
            label_value = None
            if torch.is_tensor(label_tensor):
                if label_tensor.dim() == 0:
                    scalar_label = label_tensor.item()
                    label_value = scalar_label
                else:
                    label_value = label_tensor.tolist()
            else:
                scalar_label = label_tensor
                label_value = scalar_label
            
            if label_value is None:
                label_value = label_tensor
            binary_label = 1 if (scalar_label is not None and scalar_label > 0) else label_value
            print(f"\nPatient {i}:")
            print(f"  Label: {label_value} (binary view: {binary_label})")
            print(f"  Text: ...{text_preview}")
    
    # 7. Create Data Collator
    print("\n" + "=" * 80)
    print("Creating data collator...")
    print("=" * 80)
    
    binary_classification = not multi_label_task
    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config.get('max_length'),
        binary_classification=binary_classification,
        truncation=False,
        handle_long_sequences=data_config.get('handle_long_sequences', 'warn')
    )
    print(f"  - Truncation: False (keeping full patient trajectories)")
    print(f"  - Binary classification: {binary_classification}")
    
    # 8. Train
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
        collate_fn=collate_fn
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal model saved to: {training_config['output_dir']}/final_model")
    
    return trainer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Binary Classification Fine-tuning")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    main(args.config_filepath)


