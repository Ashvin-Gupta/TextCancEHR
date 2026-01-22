import os
import pandas as pd
from src.data.unified_dataset import UnifiedEHRDataset

# CONFIG (Update paths)
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi" 
CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def check_concept_drift():
    print("Checking for Temporal Concept Leakage (e.g., COVID terms)...")
    
    # Load Validation Set with current settings (12m cutoff for cases)
    ds = UnifiedEHRDataset(
        data_dir=DATA_DIR,
        vocab_file=CONFIG_PATHS['vocab'],
        labels_file=CONFIG_PATHS['labels'],
        medical_lookup_file=CONFIG_PATHS['medical'],
        lab_lookup_file=CONFIG_PATHS['lab'],
        region_lookup_file=CONFIG_PATHS['region'],
        time_lookup_file=CONFIG_PATHS['time'],
        cutoff_months=12,
        format='text',
        split='tuning' 
    )
    
    # Terms that should ONLY appear in recent years (Controls)
    # Adjust these based on what text appears in your vocab
    future_terms = [
        "covid", "sars-cov-2", "coronavirus", 
        "pandemic", "pfizer", "astrazeneca", "telehealth",
        "video consultation", "remote consultation"
    ]
    
    case_hits = 0
    control_hits = 0
    total_cases = 0
    total_controls = 0
    
    print(f"Scanning {len(ds)} records for terms: {future_terms}...")
    
    for i in range(len(ds)):
        sample = ds[i]
        if sample is None: continue
        
        label = sample['label'].item()
        text = sample['text'].lower()
        
        has_future_term = any(term in text for term in future_terms)
        
        if label > 0:
            total_cases += 1
            if has_future_term:
                case_hits += 1
                if case_hits <= 3:
                     print(f"  [Unexpected] Case found with future term: {text[-100:]}")
        else:
            total_controls += 1
            if has_future_term:
                control_hits += 1
                
    print(f"\nRESULTS:")
    print(f"Cases with Future Terms:    {case_hits} / {total_cases} ({case_hits/total_cases:.2%})")
    print(f"Controls with Future Terms: {control_hits} / {total_controls} ({control_hits/total_controls:.2%})")
    
    if control_hits > case_hits * 10:
        print("\n>>> CONFIRMED: Temporal Leakage via Concept Drift.")
        print("    The model distinguishes Controls by seeing 'Future' concepts.")

if __name__ == "__main__":
    check_concept_drift()