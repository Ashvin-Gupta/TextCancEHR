"""
Tokenizer extension for EHR special marker tokens.

We keep this as a small, focused module so the training scripts don't have to
care about the underlying model loader details.
"""

from __future__ import annotations

import os

from unsloth import FastLanguageModel


class EHRTokenExtensionStaticTokenizer:
    """
    Extend a base LLM tokenizer with the static EHR markers used in the narratives.
    """

    TOKENS_TO_ADD = ("<TIME>", "<DEMOGRAPHIC>", "<EVENT>", "<VALUE>")

    def extend_tokenizer(self, model_name: str, max_seq_length: int = 512, load_in_4bit: bool = True):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = f"cuda:{local_rank}"

        print(f"[Rank {local_rank}] Loading model on {target_device}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map={"": local_rank},
        )

        current_vocab = tokenizer.get_vocab().keys()
        new_tokens = [t for t in self.TOKENS_TO_ADD if t not in current_vocab]
        existing_tokens = [t for t in self.TOKENS_TO_ADD if t in current_vocab]

        if existing_tokens:
            print(f"Warning: {len(existing_tokens)} tokens already exist in the tokenizer: {existing_tokens}")

        if new_tokens:
            num_added = tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added} new tokens to tokenizer: {new_tokens}")
            model.resize_token_embeddings(len(tokenizer))

            # Ensure resized embeddings land on the correct GPU for DDP jobs.
            model.get_input_embeddings().to(target_device)
            if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
                model.get_output_embeddings().to(target_device)
        else:
            print("No new tokens to add")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")

        print(f"Final vocab size: {len(tokenizer)}")
        return model, tokenizer

