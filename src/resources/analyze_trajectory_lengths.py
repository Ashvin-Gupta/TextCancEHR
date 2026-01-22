import os
import sys
import numpy as np
import pandas as pd
from src.data.unified_dataset import UnifiedEHRDataset

# --- CONFIGURATION ---
# Use the path confirmed by your previous script output
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi" 

# Standard config paths
CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def print_stats(name, data):
    if len(data) == 0:
        print(f"  {name}: No data")
        return
    
    print(f"  {name} ({len(data)} patients):")
    print(f"    Mean:   {np.mean(data):.2f}")
    print(f"    Median: {np.median(data):.2f}")
    print(f"    Std:    {np.std(data):.2f}")
    print(f"    Min:    {np.min(data)}")
    print(f"    Max:    {np.max(data)}")
    # 95th percentile to see the "long tail"
    print(f"    95th %: {np.percentile(data, 95):.2f}")

def analyze_trajectory_lengths():
    print(f"{'='*80}")
    print(f"TRAJECTORY LENGTH ANALYSIS (Cases vs Controls)")
    print(f"Cutoff: 6 Months (for Cases)")
    print(f"{'='*80}")
    
    splits = ['train', 'tuning', 'held_out']
    
    for split_name in splits:
        print(f"\nAnalyzing Split: {split_name.upper()}...")
        
        try:
            # Load dataset with classification settings
            # format='tokens' is faster than 'text' for counting length
            ds = UnifiedEHRDataset(
                data_dir=DATA_DIR,
                vocab_file=CONFIG_PATHS['vocab'],
                labels_file=CONFIG_PATHS['labels'],
                medical_lookup_file=CONFIG_PATHS['medical'],
                lab_lookup_file=CONFIG_PATHS['lab'],
                region_lookup_file=CONFIG_PATHS['region'],
                time_lookup_file=CONFIG_PATHS['time'],
                cutoff_months=6,  # The classification cutoff
                format='tokens', 
                split=split_name
            )
            
            case_lengths = []
            control_lengths = []
            
            for i in range(len(ds)):
                sample = ds[i]
                if sample is None:
                    continue
                
                # sample['tokens'] is a tensor of token IDs
                length = len(sample['tokens'])
                label = sample['label'].item()
                
                if label > 0:
                    case_lengths.append(length)
                else:
                    control_lengths.append(length)
            
            # Print Statistics
            print_stats("CASES", case_lengths)
            print("-" * 40)
            print_stats("CONTROLS", control_lengths)
            
            # Simple heuristic check
            if len(case_lengths) > 0 and len(control_lengths) > 0:
                diff_mean = np.mean(control_lengths) - np.mean(case_lengths)
                print(f"\n  >>> Gap (Control Mean - Case Mean): {diff_mean:.2f} tokens")
                if diff_mean > 50:
                    print("  [!] WARNING: Controls are significantly longer on average.")
        
        except Exception as e:
            print(f"  [!] Failed to load split {split_name}: {e}")

if __name__ == "__main__":
    analyze_trajectory_lengths()