import os
import pandas as pd
from src.data.unified_dataset import UnifiedEHRDataset

# UPDATE PATHS
DATA_DIR = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi" 
CONFIG_PATHS = {
    "vocab": "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv",
    "labels": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
    "medical": "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv",
    "lab": "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv",
    "region": "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv",
    "time": "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"
}

def check_val_leakage():
    print("Checking VALIDATION split for un-truncated cancer labels...")
    
    # Load VALIDATION set specifically
    ds = UnifiedEHRDataset(
        data_dir=DATA_DIR,
        vocab_file=CONFIG_PATHS['vocab'],
        labels_file=CONFIG_PATHS['labels'],
        medical_lookup_file=CONFIG_PATHS['medical'],
        lab_lookup_file=CONFIG_PATHS['lab'],
        region_lookup_file=CONFIG_PATHS['region'],
        time_lookup_file=CONFIG_PATHS['time'],
        cutoff_months=6,  # Your classification cutoff
        format='text',
        split='tuning'     # CHECK THE VALIDATION SET
    )
    
    leaks = 0
    nats = 0
    total_cases = 0
    
    for i in range(len(ds)):
        rec = ds.patient_records[i]
        sid = rec['subject_id']
        label = ds.subject_to_label.get(sid)
        
        if label > 0: # It's a Case
            total_cases += 1
            cancer_date = ds.subject_to_cancer_date.get(sid)
            
            # Check 1: Did date parsing fail?
            if pd.isna(cancer_date):
                nats += 1
                
            # Check 2: Is the diagnosis literally in the text?
            sample = ds[i]
            if sample is None: continue 
            
            text = sample['text'][-500:].lower() # Look at end of text
            
            # Add keywords specific to your dataset's cancer codes
            suspicious = ["malignant", "neoplasm", "cancer", "oncology", "chemotherapy"]
            found = [w for w in suspicious if w in text]
            
            if found:
                leaks += 1
                print(f"\n[!] LEAK FOUND in Val Patient {sid}")
                print(f"    Date: {cancer_date}")
                print(f"    Keywords found: {found}")
                print(f"    Text End: ...{sample['text'][-200:]}")
    
    print(f"\nSummary on VALIDATION Set:")
    print(f"Total Cases: {total_cases}")
    print(f"Missing Dates (NaT): {nats}")
    print(f"Text Leaks Detected: {leaks}")

if __name__ == "__main__":
    check_val_leakage()