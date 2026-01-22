import argparse
import yaml
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.data.unified_dataset import UnifiedEHRDataset

def create_embedding_corpus(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config['data']

    # Load PRE-COMPUTED vocabulary embeddings (FAST!)
    print("Loading pre-computed vocabulary embeddings...")
    vocab_embedding_path = data_config['vocab_embedding_path']
    token_id_to_embedding = torch.load(vocab_embedding_path)
    print(f"✓ Loaded embeddings for {len(token_id_to_embedding)} tokens")
    
    # Process each split
    splits = ['train', 'tuning', 'held_out']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Create output directory
        split_output_dir = os.path.join(data_config['embedding_output_dir'], split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Load dataset (use 'tokens' format - no need to translate!)
        dataset_args = {
            "data_dir": data_config['data_dir'],
            "vocab_file": data_config["vocab_filepath"],
            "labels_file": data_config["labels_filepath"],
            "medical_lookup_file": data_config["medical_lookup_filepath"],
            "lab_lookup_file": data_config["lab_lookup_filepath"],
            "format": "tokens", # Just get token IDs!
            "cutoff_months": data_config['cutoff_months'],
            "max_sequence_length": None
        }
        base_dataset = UnifiedEHRDataset(split=split, **dataset_args)
        
        print(f"Creating embeddings for {len(base_dataset)} patients...")
        valid_patients = 0
        
        for i in tqdm(range(len(base_dataset)), desc=f"Processing {split}"):
            item = base_dataset[i]
            if item is None:
                continue
            
            token_ids = item['tokens']  # (N,) tensor
            label = item['label']
            
            if len(token_ids) == 0:
                continue
            
            # LOOKUP pre-computed embeddings (INSTANT!)
            embeddings = torch.stack([
                token_id_to_embedding[tid.item()] 
                for tid in token_ids
            ])  # (N, 768)
            
            # Save
            output_data = {
                "embeddings": embeddings,
                "token_ids": token_ids,
                "label": label
            }
            torch.save(output_data, os.path.join(split_output_dir, f"patient_{i}.pt"))
            valid_patients += 1
        
        print(f"✓ Processed {valid_patients} patients for {split}")
    
    print(f"\n{'='*60}")
    print("Embedding corpus creation complete!")
    print(f"All splits saved to: {data_config['embedding_output_dir']}")
    print(f"Directory structure:")
    for split in splits:
        split_dir = os.path.join(data_config['embedding_output_dir'], split)
        if os.path.exists(split_dir):
            file_count = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            print(f"  - {split}/: {file_count} patient files")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    create_embedding_corpus(args.config_filepath)