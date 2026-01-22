import os
import torch
import pandas as pd
import random
from src.data.unified_dataset import UnifiedEHRDataset

# --- CONFIGURATION ---
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi" # UPDATE THIS
CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def compare_pretrain_vs_finetune(patient_limit=5):
    print("=" * 80)
    print("DEBUGGING: PRETRAINING (Memory) vs FINETUNING (Input) VIEW")
    print("=" * 80)

    # 1. Setup "Pretraining" View (What the LoRA model learned)
    # Usually cutoff is 1 month or None for pretraining
    pretrain_ds = UnifiedEHRDataset(
        data_dir=DATA_DIR,
        vocab_file=CONFIG_PATHS['vocab'],
        labels_file=CONFIG_PATHS['labels'],
        medical_lookup_file=CONFIG_PATHS['medical'],
        lab_lookup_file=CONFIG_PATHS['lab'],
        region_lookup_file=CONFIG_PATHS['region'],
        time_lookup_file=CONFIG_PATHS['time'],
        cutoff_months=1,  # The pretraining cutoff
        format='text',    # Using text format to read it easily
        split='tuning'    # Using validation split to check leakage
    )

    # 2. Setup "Finetuning/Classifer" View (What the classifier sees)
    # Usually cutoff is 12 months
    finetune_ds = UnifiedEHRDataset(
        data_dir=DATA_DIR,
        vocab_file=CONFIG_PATHS['vocab'],
        labels_file=CONFIG_PATHS['labels'],
        medical_lookup_file=CONFIG_PATHS['medical'],
        lab_lookup_file=CONFIG_PATHS['lab'],
        region_lookup_file=CONFIG_PATHS['region'],
        time_lookup_file=CONFIG_PATHS['time'],
        cutoff_months=12, # The strict classification cutoff
        format='text',
        split='tuning'    # Same split
    )

    # 3. Find Cases (Positive Labels)
    case_indices = [i for i in range(len(pretrain_ds)) 
                   if pretrain_ds.subject_to_label.get(pretrain_ds.patient_records[i]['subject_id']) > 0]
    
    selected_indices = random.sample(case_indices, min(patient_limit, len(case_indices)))

    for idx in selected_indices:
        subject_id = pretrain_ds.patient_records[idx]['subject_id']
        label = pretrain_ds.subject_to_label[subject_id]
        cancer_date = pretrain_ds.subject_to_cancer_date.get(subject_id)

        # Get the two views
        pretrain_sample = pretrain_ds[idx]
        finetune_sample = finetune_ds[idx]
        
        # Determine lengths
        pt_text = pretrain_sample['text']
        ft_text = finetune_sample['text']
        
        print(f"\n>>> PATIENT {subject_id} (Label: {label})")
        print(f"    Cancer Date: {cancer_date}")
        if pd.isna(cancer_date):
            print("    [!] WARNING: CANCER DATE IS NaT - DATA CUTOFF FAILED")

        print(f"\n    [A] PRETRAINING MEMORY (Cutoff 1m):")
        print(f"        Length: {len(pt_text)} chars")
        print(f"        End of narrative (Last 300 chars):")
        print(f"        ...{pt_text[-300:]}")

        print(f"\n    [B] CLASSIFIER INPUT (Cutoff 12m):")
        print(f"        Length: {len(ft_text)} chars")
        print(f"        End of narrative (Last 300 chars):")
        print(f"        ...{ft_text[-300:]}")

        # Check overlap
        if len(ft_text) >= len(pt_text):
            print("    [!] WARNING: Classifier input is SAME/LONGER than pretraining input. Cutoff broken.")
        else:
            diff = len(pt_text) - len(ft_text)
            print(f"    INFO: Pretraining saw {diff} more chars than Classifier.")
            
        # Check if the "Diff" contains the label
        hidden_part = pt_text[len(ft_text):]
        print(f"\n    [C] THE 'SECRET' KNOWLEDGE (Seen in Pretraining, Hidden in Classifier):")
        print(f"        ...{hidden_part[:500]}...") # Show what the model 'knows' but doesn't 'see'

if __name__ == "__main__":
    compare_pretrain_vs_finetune()