# src/data/embedded_dataset.py
import torch
import os
from torch.utils.data import Dataset
import random

class PreEmbeddedDataset(Dataset):
    def __init__(self, data_dir, split='train', task='classification', max_sequence_length=None):
        """
        Load pre-embedded data for a specific split.
        
        Args:
            data_dir: Base directory containing train/, tuning/, held_out/ subdirectories
            split: Which split to load ('train', 'tuning', 'held_out')
            task: Task type - 'classification' (encoder), 'autoregressive' (decoder), or 'both'
            max_sequence_length: Maximum sequence length for the autoregressive task
        """
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        self.task = task
        self.max_sequence_length = max_sequence_length
        self.data_files = [os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) if f.endswith('.pt')]
        self.data_files.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        # Load the pre-computed file
        data = torch.load(self.data_files[idx])
        embeddings = data['embeddings']  # (N, D_embed)
        token_ids = data['token_ids']    # (N,)
        full_seq_len = embeddings.size(0)
        
        if self.task == 'classification':
            # For encoder models: only need embeddings and label
            return {
                "embeddings": data['embeddings'],  # (N, 768)
                "label": data['label']
            }
        elif self.task == 'autoregressive':
            # We need max_sequence_length + 1 tokens for the autoregressive shift
            # (e.g., 2048 for input, 2048 for target, requires 2049 total tokens)
            target_len = self.max_sequence_length + 1
            
            if full_seq_len > target_len:
                # Find valid start indices (from 0 to N - target_len)
                max_start_idx = full_seq_len - target_len
                
                # Randomly pick a start index *only* during training
                if self.split == 'train':
                    start_idx = random.randint(0, max_start_idx)
                else:
                    # For validation, just take the last possible window
                    start_idx = max_start_idx 
                
                # Slice the window
                embeddings = embeddings[start_idx : start_idx + target_len]
                token_ids = token_ids[start_idx : start_idx + target_len]

            return {
                "embeddings": embeddings,
                "token_ids": token_ids,
                "label": data['label']
            }
        else:  # 'both'
            # Return everything
            return data