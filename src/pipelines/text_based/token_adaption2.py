import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unsloth
from unsloth import FastLanguageModel
import torch.nn.functional as F
import pandas as pd
import os
'''
This script is used to translate the EHR tokens to natural language.
It is in the format where the natural language events are split by special tokens which are then added to the tokenizer.
'''

class EHRTokenExtensionStaticTokenizer:
    """
    Class to handle extension of the tokenizer with static tokens.
    """
    
    def __init__(self):
        pass
    
    def extend_tokenizer(self, model_name, max_seq_length=512, load_in_4bit=True):
        """
        Token adaptation pipeline that extends the tokenizer with predefined tokens.
        
        Args:
            model_name: Name of the model to load
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit precision
            
        Returns:
            tuple: (model, tokenizer) with extended vocabulary
        """
        
        
        # Define tokens to add to the tokenizer
        tokens_to_add = [
            "<TIME>", "<DEMOGRAPHIC>", "<EVENT>", "<VALUE>",
        ]

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = f"cuda:{local_rank}"
        
        print(f"[Rank {local_rank}] Loading model on {target_device}...")
        
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map={"": local_rank}
        )
        
        # Get current vocabulary
        current_vocab = tokenizer.get_vocab().keys()
        print(f"Current vocab size: {len(current_vocab)}")
        
        # Filter tokens that don't already exist
        new_tokens = []
        existing_tokens = []
        
        for token in tokens_to_add:
            if token not in current_vocab:
                new_tokens.append(token)
            else:
                existing_tokens.append(token)
        
        if existing_tokens:
            print(f'Warning: {len(existing_tokens)} tokens already exist in the tokenizer')
            print(f"Existing tokens: {existing_tokens}")
        
        # Add new tokens to tokenizer
        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added} new tokens to tokenizer")
            print(f"New tokens: {new_tokens}")
            
            # Resize model embeddings to accommodate new tokens
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model embeddings to {len(tokenizer)} tokens")
            print(f"[Rank {local_rank}] Resized model embeddings.")

            # Explicitly move the resized embeddings to the correct GPU.
            # Without this, new embeddings might default to GPU 0, causing the crash.
            model.get_input_embeddings().to(target_device)
            if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
                model.get_output_embeddings().to(target_device)
            print(f"[Rank {local_rank}] Embeddings enforced to {target_device}.")
        else:
            print("No new tokens to add")
        
        # Add PAD token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        
        print(f"Final vocab size: {len(tokenizer)}")
        
        return model, tokenizer