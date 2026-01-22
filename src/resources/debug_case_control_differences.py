import os
import random
import yaml
import torch
from src.data.unified_dataset import UnifiedEHRDataset
from src.pipelines.text_based.token_adaption2 import EHRTokenExtensionStaticTokenizer

# --- CONFIGURATION ---
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi" 
MODEL_NAME = "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit" # Or your specific base model
MAX_LENGTH = 8192 # The extended context window you are using
CUTOFF_MONTHS = 12 # The classification cutoff

CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def inspect_model_inputs(split_name='tuning', num_samples=3):
    print(f"\n{'='*80}")
    print(f"DEBUGGING MODEL INPUTS (Split: {split_name})")
    print(f"Context Window: {MAX_LENGTH} tokens | Cutoff: {CUTOFF_MONTHS} months")
    print(f"{'='*80}")

    # 1. Initialize Tokenizer (The exact one used in training)
    print("Loading Tokenizer and extending with EHR vocabulary...")
    translator = EHRTokenExtensionStaticTokenizer()
    # We use a placeholder model_name just to get the base tokenizer
    # In a real run, this loads the base HF tokenizer + your added tokens
    _, tokenizer = translator.extend_tokenizer(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True 
    )
    
    # 2. Load Dataset (The Source Text)
    print(f"Loading {split_name} dataset...")
    ds = UnifiedEHRDataset(
        data_dir=DATA_DIR,
        vocab_file=CONFIG_PATHS['vocab'],
        labels_file=CONFIG_PATHS['labels'],
        medical_lookup_file=CONFIG_PATHS['medical'],
        lab_lookup_file=CONFIG_PATHS['lab'],
        region_lookup_file=CONFIG_PATHS['region'],
        time_lookup_file=CONFIG_PATHS['time'],
        cutoff_months=CUTOFF_MONTHS,
        format='text', 
        split=split_name,
        tokenizer=tokenizer
    )

    cases = []
    controls = []
    
    # Random Sampling
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    
    for idx in all_indices:
        sample = ds[idx]
        if sample is None: continue
        label = sample['label'].item()
        
        if label > 0 and len(cases) < num_samples:
            cases.append((idx, sample))
        elif label == 0 and len(controls) < num_samples:
            controls.append((idx, sample))
            
        if len(cases) >= num_samples and len(controls) >= num_samples:
            break

    def print_model_view(group_name, samples):
        print(f"\n>>> {group_name} SAMPLES (Processed by Tokenizer)")
        for i, (idx, sample) in enumerate(samples):
            rec = ds.patient_records[idx]
            subject_id = rec['subject_id']
            cancer_date = ds.subject_to_cancer_date.get(subject_id, "N/A")
            raw_text = sample['text']
            
            # --- SIMULATE TRAINING INPUT ---
            # This is exactly what the Collator does
            tokenized = tokenizer(
                raw_text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_tensors=None # Get lists
            )
            input_ids = tokenized['input_ids']
            
            # Decode back to see what the model actually attends to
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            print(f"\n  [{group_name} #{i+1}] Subject ID: {subject_id}")
            print(f"  Cancer Date: {cancer_date}")
            print(f"  Raw Text Length: {len(raw_text)} chars")
            print(f"  Token Count: {len(input_ids)} / {MAX_LENGTH}")
            
            # Check for truncation
            if len(input_ids) == MAX_LENGTH:
                print("  [!] STATUS: TRUNCATED (History lost at start/end depending on tokenizer)")
            else:
                print("  [✓] STATUS: FIT IN CONTEXT")

            print(f"  Model Input (Last 500 chars of decoded text):")
            print(f"  {'-'*40}")
            print(f"  ...{decoded_text}")
            print(f"  {'-'*40}")
            
            # Check for Future Leaks in the Decoded Text
            suspicious_years = ["2020", "2021", "2022", "2023", "covid", "pandemic"]
            found = [w for w in suspicious_years if w in decoded_text.lower()]
            if found:
                print(f"  [!] LEAK DETECTED IN MODEL INPUT: {found}")

    print_model_view("CASE", cases)
    print_model_view("CONTROL", controls)

if __name__ == "__main__":
    # Check the validation set as that's where the 99% F1 is happening
    inspect_model_inputs('tuning', num_samples=5)