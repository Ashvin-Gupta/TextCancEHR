import os
import sys
import pandas as pd
from src.data.unified_dataset import UnifiedEHRDataset

# --- CONFIGURATION (Ensure these match your actual paths) ---
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi"  # Check this path!
# DATA_DIR might need to be just "/data/home/qc25022/CancEHR-Training/data" depending on your folder structure

CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def audit_splits():
    print(f"{'='*80}")
    print(f"DATASET SPLIT AUDIT")
    print(f"{'='*80}")
    
    splits = ['train', 'tuning', 'held_out']
    split_data = {}
    
    # 1. Load and Census Each Split
    for split_name in splits:
        print(f"\nLoading split: {split_name.upper()}...")
        try:
            # Initialize dataset (using 'tokens' format is faster as we don't need text translation)
            ds = UnifiedEHRDataset(
                data_dir=DATA_DIR,
                vocab_file=CONFIG_PATHS['vocab'],
                labels_file=CONFIG_PATHS['labels'],
                medical_lookup_file=CONFIG_PATHS['medical'],
                lab_lookup_file=CONFIG_PATHS['lab'],
                region_lookup_file=CONFIG_PATHS['region'],
                time_lookup_file=CONFIG_PATHS['time'],
                cutoff_months=1, # Arbitrary, we just want IDs and Labels
                format='tokens', 
                split=split_name
            )
            
            subject_ids = set()
            cases = 0
            controls = 0
            
            for i in range(len(ds)):
                rec = ds.patient_records[i]
                sid = rec['subject_id']
                
                # Check label
                label = ds.subject_to_label.get(sid)
                
                # Add to ID set
                subject_ids.add(sid)
                
                if label is not None:
                    if label > 0:
                        cases += 1
                    else:
                        controls += 1
            
            split_data[split_name] = {
                "ids": subject_ids,
                "cases": cases,
                "controls": controls,
                "total": len(ds)
            }
            
            print(f"  -> Total: {len(ds)}")
            print(f"  -> Cases: {cases}")
            print(f"  -> Controls: {controls}")
            if len(ds) > 0:
                print(f"  -> Case Rate: {cases/len(ds):.2%}")
                
        except Exception as e:
            print(f"  [!] FAILED to load split '{split_name}': {e}")
            split_data[split_name] = {"ids": set(), "cases": 0, "controls": 0, "total": 0}

    # 2. Check for Overlaps (Leakage)
    print(f"\n{'='*80}")
    print(f"OVERLAP CHECK (LEAKAGE)")
    print(f"{'='*80}")
    
    train_ids = split_data['train']['ids']
    val_ids = split_data['tuning']['ids']
    test_ids = split_data['held_out']['ids']
    
    # Train vs Val
    tv_overlap = train_ids.intersection(val_ids)
    print(f"Train vs Validation Overlap: {len(tv_overlap)} patients")
    if len(tv_overlap) > 0:
        print(f"  [!] CRITICAL WARNING: {len(tv_overlap)} patients are in BOTH Train and Validation!")
        print(f"  Examples: {list(tv_overlap)[:5]}")

    # Train vs Test
    tt_overlap = train_ids.intersection(test_ids)
    print(f"Train vs Test Overlap:       {len(tt_overlap)} patients")
    if len(tt_overlap) > 0:
        print(f"  [!] CRITICAL WARNING: {len(tt_overlap)} patients are in BOTH Train and Test!")

    # Val vs Test
    vt_overlap = val_ids.intersection(test_ids)
    print(f"Validation vs Test Overlap:  {len(vt_overlap)} patients")
    if len(vt_overlap) > 0:
        print(f"  [!] CRITICAL WARNING: {len(vt_overlap)} patients are in BOTH Validation and Test!")

    if len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0:
        print("\n>>> SUCCESS: No patient overlap detected between splits.")
    else:
        print("\n>>> FAILURE: Patient leakage detected.")

if __name__ == "__main__":
    audit_splits()