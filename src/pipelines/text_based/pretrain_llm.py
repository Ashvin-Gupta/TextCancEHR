# src/experiments/run_llm_pretrain.py

"""
LLM Continued Pretraining Script

This script performs continued pretraining of a language model on EHR data using:
1. UnifiedEHRDataset in 'text' format for medical code translation
2. HuggingFace's ConstantLengthDataset for efficient sequence packing
3. 1-month temporal cutoff for cancer patients to avoid late-stage signals
4. Standard causal language modeling (next-token prediction)

Usage:
    python -m src.experiments.run_llm_pretrain --config_filepath configs/llm_pretrain.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
import wandb
from collections import defaultdict, Counter
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TextIteratorStreamer
)
from threading import Thread
import textwrap

import torch
from huggingface_hub import login

from src.data.unified_dataset import UnifiedEHRDataset
from src.pipelines.text_based.token_adaption2 import EHRTokenTranslator

# Custom callback to run inference after each epoch
from transformers import TrainerCallback

class InferenceCallback(TrainerCallback):
    def __init__(self, model, tokenizer, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n" + "=" * 80)
        print(f"Running inference test after epoch {state.epoch}...")
        print("=" * 80)
        run_ehr_inference(self.model, self.tokenizer, self.prompt)
        print("\n")

# Define the inference prompt
inference_prompt = '65-69GENDER FEMALEETHNICITY WHITEREGION North WestBMI: highFrailty Index score: low1dFrailty Index score: low4d-7dOphthalmological referral7d-12dFrailty Index score: low1dFrailty Index score: low2d-4dDermatitisFrailty Index score: low1d-2dDermatological referral20d-30dDermatitisContact dermatitis due to solar radiation7d-12dFrailty Index score: low20d-30dCapsulotomy of lens capsule7d-12dPsoriasis20d-30dOral administration of treatment12d-20dFrailty Index score: low7d-12dFrailty Index score: low4d-7dEnteric microscopy, culture and sensitivitiesClostridium difficile glutamate dehydrogenase immunoassay2d-4dFrailty Index score: normal7d-12dEosinophil count: normalSerum vitamin B12 level: normalAcetoacetate level: highAcute kidney injury warning stageTest request : Serum TSH level: high'


def check_tokenization_integrity(dataset, tokenizer, original_vocab_size: int, allowed_base_tokens: set):
    """
    Checks if any unexpected base model tokens (IDs < original_vocab_size) 
    are present in the tokenized training data.

    Args:
        dataset: The Hugging Face Dataset or a list of raw text narratives.
        tokenizer: The tokenizer object (with added tokens).
        original_vocab_size: The token ID that marks the start of new medical tokens.
        allowed_base_tokens: A set of token IDs for "normal", "high", "low", etc., 
                             that you *expect* to be in the base vocabulary.
    """
    unexpected_tokens_found = Counter()
    total_new_tokens = 0
    total_patients_checked = 0
    
    print(f"\n--- Running Tokenization Integrity Check ---")
    print(f"Original Vocab Size (Boundary): {original_vocab_size}")

    for i, item in enumerate(dataset):
        # Assuming item is the raw text string from your dataset
        if isinstance(item, dict) and 'text' in item:
            text = item['text']
        elif isinstance(item, str):
            text = item
        else:
            continue
            
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_patients_checked += 1
        
        # 1. Check for tokens below the threshold
        for token_id in token_ids:
            if token_id >= original_vocab_size:
                total_new_tokens += 1
                continue
            
            # Token ID is in the original base vocabulary range
            if token_id not in allowed_base_tokens:
                # We found an UNEXPECTED token from the original model's vocabulary
                decoded_token = tokenizer.decode([token_id])
                unexpected_tokens_found[f"ID {token_id}: '{decoded_token}'"] += 1

        if total_patients_checked % 1000 == 0 and total_patients_checked > 0:
            print(f"  - Checked {total_patients_checked} patients...")
            
    # --- Report Results ---
    print("-" * 80)
    print(f"CHECK COMPLETE. Total Patients Checked: {total_patients_checked}")
    print(f"Total tokens from new vocabulary (ID >= {original_vocab_size}): {total_new_tokens}")
    
    if unexpected_tokens_found:
        print("\n🚨 CRITICAL ERROR: Found UNEXPECTED BASE MODEL TOKENS in Training Data:")
        for token_str, count in unexpected_tokens_found.most_common(30):
            print(f"  - {token_str}: Found {count} times.")
        
        print("\nThese tokens are likely causing the model collapse (e.g., '2', 'word count').")
        print("ACTION: You must sanitize your data to ensure these base tokens are replaced by a single, new medical event token, or ensure the tokenizer is explicitly told to tokenize them as expected.")
        return False
    else:
        print("\n✅ Tokenization Check Passed: No unexpected base model tokens found.")
        return True


def extract_text(base_dataset, tokenizer):
        """Extracts all valid text narratives and adds EOS token."""
        text_list = []
        # Use eos_token if it exists, otherwise use an empty string
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        
        print(f"  - Processing {len(base_dataset)} patients...")
        # We iterate through the base_dataset to get the text
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item is not None:
                # item['text'] is the narrative from UnifiedEHRDataset
                text_list.append(item['text'] + eos_token)
        print(f"  - Extracted {len(text_list)} valid narratives.")
        return text_list

def verify_patient(train_text_list, tokenizer):
    print("\nVerifying data - First 3 patient narratives:")
    for i in range(min(3, len(train_text_list))):
        print(f"\n--- PATIENT {i} ---")
        # Print the first 1000 chars
        print(f"{train_text_list[i][:1000]}...")
        
        # Tokenize and show token analysis
        print(f"\n--- PATIENT {i} TOKENIZATION ---")
        tokens = tokenizer.tokenize(train_text_list[i])
        token_ids = tokenizer.encode(train_text_list[i], add_special_tokens=False)
        
        print(f"Text length: {len(train_text_list[i])} characters")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Number of token IDs: {len(token_ids)}")
        print(f"First 100 tokens: {tokens[:100]}")
        print(f"First 100 token IDs: {token_ids[:100]}")
        
        # Show token-to-text mapping for first few tokens
        print(f"Token-to-text mapping (first 10):")
        for j in range(min(100, len(tokens))):
            decoded = tokenizer.decode([token_ids[j]])
            print(f"  Token {j}: '{tokens[j]}' -> ID {token_ids[j]} -> Decoded: '{decoded}'")

    print("\n" + "=" * 80)
    print("Creating SFT datasets...")
    print("=" * 80)

def run_ehr_inference(model, tokenizer, prompt: str, max_new_tokens: int = 256, max_print_width: int = 100):
    """
    Runs causal language modeling inference on the fine-tuned model and displays
    the raw stream and formatted event output.
    
    Args:
        model: The fine-tuned LoRA model (e.g., Unsloth FastLanguageModel).
        tokenizer: The corresponding tokenizer.
        prompt: The input medical event sequence.
        max_new_tokens: Maximum number of tokens to generate.
        max_print_width: Width for text wrapping the final output.
    """
    
    # 1. Setup for Inference
    # The call to for_inference() is correct for Unsloth models
    model.eval()
    FastLanguageModel.for_inference(model)

    print('First checking token 2')
    token_id_2 = tokenizer.encode('2', add_special_tokens=False)
    print(f"Token ID for '2': {token_id_2}")

    # Encode the prompt and move to the correct device (assuming CUDA is available)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print("-" * 80)
    print(f"PROMPT:\n{textwrap.fill(prompt, width=max_print_width)}\n")
    print("MODEL GENERATION:")
    print("-" * 80)

    # Setup the streamer
    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    all_generated_chunks = []
    
    # 2. Setup Generation Arguments (Critical for unstable models)
    # The key change is adding do_sample=False to force greedy decoding
    generation_kwargs = dict(
        inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        # --- CRITICAL DECODING PARAMETERS ---
        do_sample=True,
        temperature=0.6,
        top_p = 0.9,
        top_k = 40,
        repetition_penalty = 1.1,
        num_beams=1,     
        pad_token_id=tokenizer.eos_token_id, # Safest default for Causal LM
        eos_token_id=tokenizer.eos_token_id, # Ensure generation stops on EOS
        # ------------------------------------
    )
    
    # 3. Start Generation Thread and Stream Raw Output
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("MODEL OUTPUT (RAW STREAM):")
    for new_text in text_streamer:
        print(new_text, end="")
        all_generated_chunks.append(new_text)

    # 4. Wait and Format Output
    thread.join()

    print("\n\n" + "=" * 80)
    print("--- Formatted Readable Event Sequence ---")
    print("=" * 80)

    # Combine all chunks
    full_generated_text = "".join(all_generated_chunks)
    
    # Get all the token IDs that were generated
    generated_token_ids = tokenizer.encode(full_generated_text, add_special_tokens=False)
    
    # Decode each token individually to get the event strings (better for event tokens)
    event_tokens = []
    for token_id in generated_token_ids:
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        # Only include non-empty tokens
        if decoded_token:
            event_tokens.append(decoded_token)
    
    # Join with spaces and use textwrap.fill for readability
    readable_output = ", ".join(event_tokens)
    
    if not readable_output:
        print("GENERATION COLLAPSED (Model predicted EOS/PAD immediately).")
    else:
        print(textwrap.fill(readable_output, width=max_print_width))
    print("\n")

def main(config_path: str):
    """
    Main function to run LLM continued pretraining.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # 1. Load Config
    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set up WandB and hugging face token
    wandb_config = config.get('wandb', {})
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    lora_config = config['lora']

    # Build default run name from hyperparameters
    model_name = model_config['model_name'].split('/')[-1]
    default_run_name = (
        f"{model_name}-pretrain-{data_config['cutoff_months']}month-cutoff"
        f"_r{lora_config['r']}"
        f"_alpha{lora_config['lora_alpha']}"
        f"_dropout{lora_config['lora_dropout']}"
        f"_lr{training_config['learning_rate']}"
        f"_bs{training_config['batch_size']}"
        f"_wd{training_config['weight_decay']}"
        f"_ga{training_config['gradient_accumulation_steps']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(default_run_name)

    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "ehr-llm-pretraining")
        run_name = wandb_config.get("run_name", default_run_name)
        report_to = "wandb"
        
        wandb.init(
            project=wandb_config.get("project", "ehr-llm-pretraining"), 
            config=config, # Pass entire YAML as defaults
            name=run_name
        )
    else:
        run_name = default_run_name
        report_to = "none"
    
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                line = f.readline().strip()
                hf_token = line.split('=')[1].strip('"')
                if hf_token:
                    print(f"Loaded HuggingFace token from {token_file}: {hf_token}")
        except Exception as e:
            print(f"Failed to read token from {token_file}: {e}")
    else:
        print(f"No API keys file found at {token_file}")
    
    print(f"Run name: {run_name}")
    
    login(token=str(hf_token))

    # 3. Load Model with Unsloth
    print("\n" + "=" * 80)
    print(f"Loading model: {model_config['model_name']}")
    print("=" * 80)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print(f"  - CUDA not available, using CPU")
    
    translator = EHRTokenTranslator(data_config["medical_lookup_filepath"], data_config["lab_lookup_filepath"], data_config["region_lookup_filepath"])
    unique_concepts = translator.extract_translated_concepts(data_config["vocab_filepath"])
    # print(f"Unique concepts: {unique_concepts}")
    
    # Perform token adaptation BEFORE applying LoRA
    model, tokenizer = translator.token_adaptation(
        original_model_name=model_config['base_model_name'],
        unsloth_model_name=model_config['model_name'],
        new_concepts=unique_concepts,
        max_seq_length=model_config['max_length'],  # Pass the max_length from config
        load_in_4bit=training_config.get('load_in_4bit', True)  # Pass load_in_4bit from config
    )

    # 4. Create Base Datasets (text format)
    print("\n" + "=" * 80)
    print("Creating datasets in 'text' format...")
    print("=" * 80)
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "format": 'text',  # Use existing text format!
        "cutoff_months": data_config.get("cutoff_months", 1),  # Default 1-month cutoff
        "max_sequence_length": None  # No truncation - we'll pack sequences
    }
    
    print("\nLoading training data...")
    train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    print(f"  - Loaded {len(train_base_dataset)} training patients")
    
    print("\nLoading validation data...")
    val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  - Loaded {len(val_base_dataset)} validation patients")
    

    print("\n" + "=" * 80)
    print("Verifying data - First 3 patient narratives:")
    print("=" * 80)
    
    # 5. Extract text from datasets
    train_text_list = extract_text(train_base_dataset, tokenizer)
    val_text_list = extract_text(val_base_dataset, tokenizer)

    # Verify the data
    verify_patient(train_text_list, tokenizer)

    train_dataset = Dataset.from_dict({"text": train_text_list})
    val_dataset = Dataset.from_dict({"text": val_text_list})

    V_orig = 151669
    allowed_ids = {}
    check_tokenization_integrity(train_dataset, tokenizer, V_orig, allowed_ids)

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.get('r', 16),
        target_modules = lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]), # Modules for Mistral/Llama
        lora_alpha = lora_config.get('lora_alpha', 16),
        lora_dropout = lora_config.get('lora_dropout', 0.05),
        bias = lora_config.get('bias', "none"),
        use_gradient_checkpointing = training_config.get('gradient_checkpointing', 'unsloth'),
        random_state = 42,
        use_rslora = lora_config.get('use_rslora', True),
        loftq_config = lora_config.get('loftq_config', None),

    )
    print("  - Applied LoRA adapters (PEFT) to the model.")

    # 6. Set Up Training Arguments
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    
    training_args = TrainingArguments(
        dataloader_num_workers=training_config.get('dataloader_num_workers', 8),

        output_dir=training_config['output_dir'],
        overwrite_output_dir=training_config.get('overwrite_output_dir', True),
        
        # Training hyperparameters
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=training_config.get('warmup_steps', 500),
        
        # Logging and evaluation
        logging_steps=training_config.get('logging_steps', 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500),
        
        # Saving
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 1000),
        save_total_limit=training_config.get('save_total_limit', 2),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        
        # Performance
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        
        # Reporting
        report_to=report_to,
        run_name=run_name,
        
        # Other
        remove_unused_columns=False,
    )
    
    print(f"  - Output directory: {training_args.output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  - FP16: {training_args.fp16}, BF16: {training_args.bf16}")
    

    # 7. Create SFTTrainer with inference callback
    print("\nInitializing SFTTrainer...")
    inference_callback = InferenceCallback(model, tokenizer, inference_prompt)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text", # Key from our TextDatasetForSFT
        max_seq_length = model_config['max_length'],
        args = training_args,
        packing = True, # --- THIS IS THE EFFICIENT PACKING! ---
        callbacks = [inference_callback],
    )
    
    # 10. Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 11. Save Final Model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    
    print(f"  - Model saved to: {final_model_path}")
    
    # 12. Final Evaluation
    if val_dataset:
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        
        eval_results = trainer.evaluate()
        print("\nValidation Results:")
        for key, value in eval_results.items():
            print(f"  - {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # 13. Run Final Inference
    print("\n" + "=" * 80)
    print("Running final inference test...")
    print("=" * 80)

    print(f"PROMPT: {inference_prompt}\n")
    run_ehr_inference(model, tokenizer, inference_prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Continued Pretraining on EHR Data")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config_filepath)