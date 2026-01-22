# src/data/embedding_collator.py

"""
Collate functions for pre-embedded EHR data.

Provides two collate functions:
1. classification_collate_fn: For encoder models doing classification
2. autoregressive_collate_fn: For decoder models doing next-event prediction
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def classification_collate_fn(batch):
    """
    Collate function for classification tasks with pre-embedded data.
    
    Pads embeddings to the maximum sequence length in the batch and creates
    a padding mask to identify valid positions.
    
    Args:
        batch: List of dicts with keys:
            - embeddings: (N_i, 768) tensor
            - label: scalar tensor
    
    Returns:
        dict with:
            - embeddings: (batch_size, max_seq_len, 768) padded tensor
            - padding_mask: (batch_size, max_seq_len) boolean mask (True = valid)
            - labels: (batch_size,) tensor
        
        Returns None if batch is empty after filtering.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    embeddings_list = [item['embeddings'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad embeddings to longest sequence in batch
    # pad_sequence expects List[Tensor] where each tensor is (seq_len, features)
    padded_embeddings = pad_sequence(
        embeddings_list, 
        batch_first=True, 
        padding_value=0.0
    )  # (batch_size, max_seq_len, 768)
    
    # Create padding mask (True = valid token, False = padding)
    seq_lengths = torch.tensor([emb.size(0) for emb in embeddings_list])
    max_len = padded_embeddings.size(1)
    
    # Create mask: True for positions < seq_length, False for padding
    padding_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    # Shape: (batch_size, max_seq_len)
    
    return {
        'embeddings': padded_embeddings,       # (B, max_N, 768)
        'padding_mask': padding_mask,          # (B, max_N)
        'labels': labels                       # (B,)
    }


def autoregressive_collate_fn(batch):
    """
    Collate function for autoregressive (next-event prediction) tasks.
    
    Pads both embeddings and token_ids, creates input/target pairs for
    next-token prediction, and generates padding masks.
    
    Args:
        batch: List of dicts with keys:
            - embeddings: (N_i, 768) tensor
            - token_ids: (N_i,) tensor
            - label: scalar tensor
    
    Returns:
        dict with:
            - input_embeddings: (batch_size, max_seq_len-1, 768) input sequence
            - target_token_ids: (batch_size, max_seq_len-1) target tokens
            - padding_mask: (batch_size, max_seq_len-1) boolean mask (True = valid)
            - labels: (batch_size,) tensor (for optional classification head)
        
        Returns None if batch is empty after filtering.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    embeddings_list = [item['embeddings'] for item in batch]
    token_ids_list = [item['token_ids'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Pad both embeddings and token_ids to longest sequence in batch
    padded_embeddings = pad_sequence(
        embeddings_list, 
        batch_first=True, 
        padding_value=0.0
    )  # (batch_size, max_seq_len, 768)
    
    padded_token_ids = pad_sequence(
        token_ids_list, 
        batch_first=True, 
        padding_value=0
    )  # (batch_size, max_seq_len)
    
    # Create padding mask
    seq_lengths = torch.tensor([emb.size(0) for emb in embeddings_list])
    max_len = padded_embeddings.size(1)
    padding_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    # Shape: (batch_size, max_seq_len)
    
    # Create input/target pairs for autoregressive training
    # Input: embeddings[:-1], Target: token_ids[1:]
    # This way the model learns to predict the next token given previous embeddings
    input_embeddings = padded_embeddings[:, :-1, :]  # (B, max_N-1, 768)
    target_token_ids = padded_token_ids[:, 1:]       # (B, max_N-1)
    input_mask = padding_mask[:, :-1]                # (B, max_N-1)
    
    return {
        'input_embeddings': input_embeddings,   # (B, max_N-1, 768)
        'target_token_ids': target_token_ids,   # (B, max_N-1)
        'padding_mask': input_mask,             # (B, max_N-1)
        'labels': labels                        # (B,) - for optional classification head
    }


def get_collate_fn(task):
    """
    Helper function to get the appropriate collate function based on task.
    
    Args:
        task: 'classification' or 'autoregressive'
    
    Returns:
        Appropriate collate function
    """
    if task == 'classification':
        return classification_collate_fn
    elif task == 'autoregressive':
        return autoregressive_collate_fn
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'autoregressive'")

