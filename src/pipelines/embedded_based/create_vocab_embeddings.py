# NEW SCRIPT: src/experiments/create_vocabulary_embeddings.py

import argparse
import yaml
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

def create_vocabulary_embeddings(config_path):
    """
    Create E5 embeddings for the entire vocabulary (one-time operation).
    This is much faster than embedding each patient's events separately.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    model_config = config['model']
    
    # Load E5 model
    print(f"Loading {model_config['model_name']} embedding model...")
    device = model_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    embed_model = SentenceTransformer(model_config['model_name'], device=device)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab_df = pd.read_csv(data_config['vocab_filepath'])
    id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()
    
    # Load medical and lab lookups
    medical_df = pd.read_csv(data_config['medical_lookup_filepath'])
    medical_lookup = pd.Series(
        medical_df['term'].values, 
        index=medical_df['code'].astype(str).str.upper()
    ).to_dict()
    
    lab_df = pd.read_csv(data_config['lab_lookup_filepath'])
    lab_lookup = pd.Series(
        lab_df['term'].values, 
        index=lab_df['code'].astype(str).str.upper()
    ).to_dict()
    
    # Translate function (same as UnifiedEHRDataset._translate_token)
    def translate_token(token_string):
        if not isinstance(token_string, str):
            return ""
        
        if token_string.startswith('DIAG_') or token_string.startswith('PROC_'):
            code = token_string.split('_', 1)[1]
            return medical_lookup.get(code.upper(), token_string)
        elif token_string.startswith('LAB_'):
            code = token_string.split('_', 1)[1]
            return lab_lookup.get(code.upper(), token_string)
        elif token_string.startswith('MED_'):
            return token_string.replace('MED_', 'Medication: ')
        else:
            return token_string
    
    # Create embeddings for all unique vocabulary tokens
    print(f"\nEmbedding {len(id_to_token_map)} vocabulary tokens...")
    
    token_id_to_embedding = {}
    
    for token_id, token_string in tqdm(id_to_token_map.items()):
        # Translate to natural language
        text = translate_token(token_string)
        
        if text:  # Only embed if translation exists
            # Embed using E5
            embedding = embed_model.encode(text, convert_to_tensor=True, device=device)
            token_id_to_embedding[token_id] = embedding.cpu()
        else:
            # Use zero vector for empty translations
            token_id_to_embedding[token_id] = torch.zeros(768)
    
    # Save the lookup table
    output_path = data_config['vocab_embedding_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving vocabulary embeddings to: {output_path}")
    torch.save(token_id_to_embedding, output_path)
    
    print(f"✓ Created embeddings for {len(token_id_to_embedding)} vocabulary tokens")
    print(f"  Embedding dimension: {token_id_to_embedding[0].shape}")
    
    return token_id_to_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True)
    args = parser.parse_args()
    
    create_vocabulary_embeddings(args.config_filepath)