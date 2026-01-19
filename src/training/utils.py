import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import yaml
import math
import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed
from torch.optim.lr_scheduler import LambdaLR


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

def compute_and_sort_by_length(dataset, tokenizer, shuffle_buckets=True, num_buckets=10):
    """Sort dataset by length with optional bucketed shuffling."""
    print(f"  - Computing lengths for {len(dataset)} samples...")
    
    # Compute lengths
    lengths = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is not None and 'text' in sample:
            length = len(tokenizer.encode(sample['text'], add_special_tokens=True))
            lengths.append((idx, length))
        else:
            lengths.append((idx, 0))
    
    # Sort by length
    lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in lengths]
    
    # Optional: Shuffle within buckets to maintain some randomness
    if shuffle_buckets:
        bucket_size = len(sorted_indices) // num_buckets
        bucketed_indices = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < num_buckets - 1 else len(sorted_indices)
            bucket = sorted_indices[start:end]
            np.random.shuffle(bucket)
            bucketed_indices.extend(bucket)
        sorted_indices = bucketed_indices
    
    # Create sorted dataset
    sorted_dataset = torch.utils.data.Subset(dataset, sorted_indices)
    
    lengths_only = [l for _, l in lengths]
    print(f"  - Length range: {min(lengths_only)} to {max(lengths_only)} tokens")
    print(f"  - Mean length: {np.mean(lengths_only):.0f} tokens")
    
    return sorted_dataset

def load_LoRA_model(config: dict):
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    print("\n" + "=" * 80)
    print(f"Loading pretrained model from: {model_config['pretrained_checkpoint']}")
    print("=" * 80)

    # STEP A: Explicitly load the correct Base Model (Qwen 2.5)
    # We force the model_name to be the base model, NOT the checkpoint path yet.
    base_model_name = model_config['unsloth_model']
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name, 
        max_seq_length=data_config['max_length'], # Needs to be higher than the max length in the data
        dtype=None,
        load_in_4bit=training_config.get('load_in_4bit', True),
    )
    print(f'Original tokenizer size: {len(tokenizer)}')

    # STEP B: Load the tokenizer from your checkpoint to get the new vocab size
    # This ensures we have the 151673 size including your 4 special tokens
    checkpoint_tokenizer = AutoTokenizer.from_pretrained(model_config['pretrained_checkpoint'])
    print(f'Checkpoint tokenizer size: {len(checkpoint_tokenizer)}')
    # Replace the standard tokenizer with your extended one
    tokenizer = checkpoint_tokenizer
    print(f'New tokenizer size: {len(tokenizer)}')
    # STEP C: Resize the model embeddings to match the checkpoint (151673)
    model.resize_token_embeddings(len(tokenizer))
    
    # STEP D: Load the adapters (PeftModel)
    # Since FastLanguageModel wraps the model, we access the internal model to load adapters if needed,
    # but usually, we can just load the adapter on top.
    model.load_adapter(model_config['pretrained_checkpoint'])

    print(f"  - Loaded model with {len(tokenizer)} tokens in vocabulary")
    print(f"  - Model type: {type(model).__name__}")
    return model, tokenizer
