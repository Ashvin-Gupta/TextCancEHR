# src/pipelines/text_based/feature_ablation_analysis.py

"""
Feature Ablation Analysis for LLM Classifier

Measures feature importance by:
1. Running inference on complete patient record (baseline)
2. Systematically removing medical events/tokens
3. Re-running inference and measuring prediction change
4. Features that cause large drops in cancer probability = important

This is complementary to gradient-based methods and doesn't require retraining.
"""

import argparse
import yaml
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

from src.pipelines.text_based.interpretability import (
    load_classifier_for_analysis,
    WeightAnalysis,
)
from src.data.unified_dataset import UnifiedEHRDataset
from src.training.utils import seed_all


def get_medical_events_from_text(text: str) -> List[Tuple[int, int, str]]:
    """
    Extract medical events from text with their positions.
    
    A medical event is defined as text between semicolons (or sentence boundaries).
    
    Returns:
        List of (start_pos, end_pos, event_text) tuples
    """
    events = []
    
    # Split on semicolons which separate events in your EHR text
    parts = text.split(';')
    current_pos = 0
    
    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            start = text.find(part_stripped, current_pos)
            end = start + len(part_stripped)
            events.append((start, end, part_stripped))
            current_pos = end
    
    return events


def load_medical_terms(medical_dict_path: str, max_terms: int = 382) -> Dict[str, str]:
    """
    Load medical terms from MedicalDictTranslation2.csv.
    
    Args:
        medical_dict_path: Path to MedicalDictTranslation2.csv
        max_terms: Number of terms of interest (first N rows)
    
    Returns:
        Dictionary mapping term text to code
    """
    df = pd.read_csv(medical_dict_path)
    
    # First max_terms are terms of interest
    terms_of_interest = df.head(max_terms)
    
    # Create mapping from term text to code
    term_to_code = {}
    for _, row in terms_of_interest.iterrows():
        code = row['code']
        term = row['term']
        # Skip NaN or non-string terms
        if isinstance(term, str):
            term_to_code[term.lower()] = code
    
    return term_to_code


def categorize_medical_term(term_text: str) -> str:
    """
    Categorize a medical term based on its content.
    
    Returns:
        Category: 'symptom', 'diagnosis', 'medication', 'lab', 'lifestyle', 'demographic', 'other'
    """
    term_lower = term_text.lower()
    
    # Symptoms
    if any(word in term_lower for word in ['pain', 'ache', 'fatigue', 'nausea', 'vomiting', 
                                             'bleeding', 'fever', 'cough', 'dyspnoea', 'syncope']):
        return 'symptom'
    
    # Diagnoses/Conditions
    if any(word in term_lower for word in ['disease', 'syndrome', 'disorder', 'cancer', 'carcinoma',
                                             'neoplasia', 'tumor', 'itis', 'osis', 'pathy', 'emia']):
        return 'diagnosis'
    
    # Medications
    if any(word in term_lower for word in ['drug', 'medication', 'therapy', 'statin', 'prazole',
                                             'metformin', 'insulin', 'antibiotic', 'aspirin']):
        return 'medication'
    
    # Labs/Measurements
    if any(word in term_lower for word in ['albumin', 'bilirubin', 'creatinine', 'glucose', 
                                             'hemoglobin', 'platelet', 'cholesterol', 'blood']):
        return 'lab'
    
    # Lifestyle
    if any(word in term_lower for word in ['smoker', 'drinker', 'alcohol', 'tobacco', 'exercise']):
        return 'lifestyle'
    
    # Demographics
    if any(word in term_lower for word in ['age', 'gender', 'ethnicity', 'race', 'birth']):
        return 'demographic'
    
    return 'other'


def get_medical_concepts_from_text(
    text: str, 
    medical_terms: Dict[str, str]
) -> List[Tuple[int, int, str, str, str]]:
    """
    Extract specific medical concepts from text using the medical dictionary.
    
    Args:
        text: Patient text narrative
        medical_terms: Dictionary mapping term text (lowercase) to code
    
    Returns:
        List of (start_pos, end_pos, concept_text, concept_code, concept_category) tuples
    """
    concepts = []
    text_lower = text.lower()
    
    # Search for each medical term in the text
    for term_text, term_code in medical_terms.items():
        # Find all occurrences of this term
        start_pos = 0
        while True:
            pos = text_lower.find(term_text, start_pos)
            if pos == -1:
                break
            
            # Get the actual text (preserving case)
            end_pos = pos + len(term_text)
            actual_text = text[pos:end_pos]
            
            # Categorize the term
            category = categorize_medical_term(term_text)
            
            concepts.append((pos, end_pos, actual_text, term_code, category))
            start_pos = end_pos
    
    # Sort by position
    concepts.sort(key=lambda x: x[0])
    
    return concepts


def remove_text_span(text: str, start: int, end: int, replacement: str = "") -> str:
    """Remove a span of text and optionally replace with something else."""
    return text[:start] + replacement + text[end:]


def run_inference(model, tokenizer, text: str, device: str = "cuda") -> Tuple[float, float]:
    """
    Run inference on text and return prediction.
    
    Returns:
        (cancer_probability, log_odds)
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=12000,
        padding=False
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits'][0].cpu().numpy()  # (2,)
        
        # Convert to probability
        log_odds = logits[1] - logits[0]
        prob_cancer = 1 / (1 + np.exp(-log_odds))
    
    return float(prob_cancer), float(log_odds)


def analyze_event_importance(
    model,
    tokenizer,
    text: str,
    patient_id: int,
    true_label: int,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Analyze importance of each medical event by ablation.
    
    Returns:
        DataFrame with columns: event_idx, event_text, baseline_prob, ablated_prob, 
                                importance_score (change in probability)
    """
    # Get baseline prediction
    baseline_prob, baseline_logodds = run_inference(model, tokenizer, text, device)
    
    # Extract medical events
    events = get_medical_events_from_text(text)
    
    results = []
    
    print(f"  Analyzing {len(events)} medical events...")
    
    for i, (start, end, event_text) in enumerate(events):
        # Remove this event and re-run inference
        text_ablated = remove_text_span(text, start, end, replacement="")
        
        try:
            ablated_prob, ablated_logodds = run_inference(model, tokenizer, text_ablated, device)
            
            # Importance = how much removing this event changes the prediction
            importance = baseline_prob - ablated_prob  # Positive = removing decreases cancer prob
            logodds_change = baseline_logodds - ablated_logodds
            
            results.append({
                'patient_id': patient_id,
                'true_label': true_label,
                'event_idx': i,
                'event_text': event_text[:100],  # Truncate for display
                'baseline_prob': baseline_prob,
                'ablated_prob': ablated_prob,
                'importance_score': importance,
                'logodds_change': logodds_change,
                'event_length': len(event_text)
            })
        except Exception as e:
            print(f"    Error processing event {i}: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df


def analyze_concept_importance(
    model,
    tokenizer,
    text: str,
    patient_id: int,
    true_label: int,
    medical_terms: Dict[str, str],
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Analyze importance of specific medical concepts by ablation.
    
    Args:
        medical_terms: Dictionary from load_medical_terms()
    
    Returns:
        DataFrame with concept-level importance scores
    """
    # Get baseline prediction
    baseline_prob, baseline_logodds = run_inference(model, tokenizer, text, device)
    
    # Extract medical concepts
    concepts = get_medical_concepts_from_text(text, medical_terms)
    
    results = []
    
    print(f"  Analyzing {len(concepts)} medical concepts...")
    
    for i, (start, end, concept_text, concept_code, concept_category) in enumerate(concepts):
        # Remove this concept and re-run inference
        text_ablated = remove_text_span(text, start, end, replacement="")
        
        try:
            ablated_prob, ablated_logodds = run_inference(model, tokenizer, text_ablated, device)
            
            importance = baseline_prob - ablated_prob
            logodds_change = baseline_logodds - ablated_logodds
            
            results.append({
                'patient_id': patient_id,
                'true_label': true_label,
                'concept_idx': i,
                'concept_text': concept_text,
                'concept_code': concept_code,
                'concept_category': concept_category,
                'baseline_prob': baseline_prob,
                'ablated_prob': ablated_prob,
                'importance_score': importance,
                'logodds_change': logodds_change,
            })
        except Exception as e:
            print(f"    Error processing concept {i}: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df


def visualize_importance_results(
    event_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    output_dir: str,
    patient_id: int
) -> None:
    """Create visualizations of feature importance."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top/Bottom Important Events
    ax1 = axes[0, 0]
    top_events = event_df.nlargest(15, 'importance_score')
    ax1.barh(range(len(top_events)), top_events['importance_score'], color='crimson', alpha=0.7)
    ax1.set_yticks(range(len(top_events)))
    ax1.set_yticklabels([e[:40] + '...' if len(e) > 40 else e for e in top_events['event_text']], fontsize=8)
    ax1.set_xlabel('Importance Score (ΔP)', fontsize=11)
    ax1.set_title(f'Top 15 Most Important Events (Removing → ↓ Cancer Prob)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 2. Events that Increase Cancer Probability when Removed
    ax2 = axes[0, 1]
    bottom_events = event_df.nsmallest(15, 'importance_score')
    ax2.barh(range(len(bottom_events)), bottom_events['importance_score'], color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(bottom_events)))
    ax2.set_yticklabels([e[:40] + '...' if len(e) > 40 else e for e in bottom_events['event_text']], fontsize=8)
    ax2.set_xlabel('Importance Score (ΔP)', fontsize=11)
    ax2.set_title(f'Events that Suppress Cancer Signal (Removing → ↑ Cancer Prob)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # 3. Concept Category Importance
    if not concept_df.empty:
        ax3 = axes[1, 0]
        concept_importance = concept_df.groupby('concept_category')['importance_score'].agg(['mean', 'count', 'std'])
        concept_importance = concept_importance.sort_values('mean', ascending=False)
        
        bars = ax3.bar(range(len(concept_importance)), concept_importance['mean'], 
                      color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(concept_importance)))
        ax3.set_xticklabels(concept_importance.index, rotation=45, ha='right')
        ax3.set_ylabel('Average Importance Score', fontsize=11)
        ax3.set_title('Average Importance by Concept Category', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (idx, row) in enumerate(concept_importance.iterrows()):
            ax3.text(i, row['mean'], f"n={int(row['count'])}", 
                    ha='center', va='bottom', fontsize=8)
    
    # 4. Distribution of Importance Scores
    ax4 = axes[1, 1]
    ax4.hist(event_df['importance_score'], bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax4.set_xlabel('Importance Score (ΔP)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Event Importance Scores', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ablation_analysis_patient_{patient_id}.png'), dpi=150)
    plt.close()


def main(config_filepath: str, checkpoint_path: str, output_dir: str, num_samples: int = 5, 
         medical_dict_path: str = None):
    """Main ablation analysis pipeline."""
    seed_all(42)
    
    print("\n" + "=" * 80)
    print("FEATURE ABLATION ANALYSIS")
    print("=" * 80)
    print(f"\nConfig:     {config_filepath}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    
    # 1. Load model
    model, tokenizer, config = load_classifier_for_analysis(
        config_path=config_filepath,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # 1.5. Load medical terms
    if medical_dict_path is None:
        # Default path
        medical_dict_path = "src/resources/MedicalDictTranslation2.csv"
    
    print(f"\nLoading medical dictionary from: {medical_dict_path}")
    medical_terms = load_medical_terms(medical_dict_path, max_terms=382)
    print(f"  Loaded {len(medical_terms)} terms of interest")
    
    # 2. Load dataset
    print("\nLoading dataset...")
    data_config = config['data']
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "time_lookup_file": data_config["time_lookup_filepath"],
        "format": 'text',
        "cutoff_months": data_config.get("cutoff_months", 1),
        "max_sequence_length": None,
        "tokenizer": None,
        "data_type": data_config.get('data_type', 'raw')
    }
    
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  Loaded {len(val_dataset)} validation samples")
    
    # 3. Find cancer and control patients
    cancer_indices = []
    control_indices = []
    
    for i in range(min(len(val_dataset), 500)):
        sample = val_dataset[i]
        if sample is not None:
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            if label > 0:
                cancer_indices.append(i)
            else:
                control_indices.append(i)
    
    print(f"  Found {len(cancer_indices)} cancer patients")
    print(f"  Found {len(control_indices)} control patients")
    
    # 4. Analyze selected patients
    import random
    sample_cancer = random.sample(cancer_indices, min(num_samples, len(cancer_indices)))
    
    all_event_results = []
    all_concept_results = []
    
    print("\n" + "=" * 80)
    print("ANALYZING CANCER PATIENTS")
    print("=" * 80)
    
    for idx in sample_cancer:
        sample = val_dataset[idx]
        text = sample['text']
        label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
        
        print(f"\n{'='*70}")
        print(f"Patient {idx} (True Label: {'Cancer' if label > 0 else 'Control'})")
        print(f"{'='*70}")
        print(f"Text length: {len(text)} characters")
        
        # Analyze events
        event_df = analyze_event_importance(model, tokenizer, text, idx, label, device)
        all_event_results.append(event_df)
        
        # Analyze concepts (using medical dictionary)
        concept_df = analyze_concept_importance(model, tokenizer, text, idx, label, medical_terms, device)
        all_concept_results.append(concept_df)
        
        # Print top results
        print(f"\n  Top 5 Most Important Events (removing → ↓ cancer prob):")
        for i, row in event_df.nlargest(5, 'importance_score').iterrows():
            print(f"    {row['importance_score']:+.4f}: {row['event_text'][:80]}...")
        
        print(f"\n  Top 5 Suppressive Events (removing → ↑ cancer prob):")
        for i, row in event_df.nsmallest(5, 'importance_score').iterrows():
            print(f"    {row['importance_score']:+.4f}: {row['event_text'][:80]}...")
        
        # Visualize
        visualize_importance_results(event_df, concept_df, output_dir, idx)
    
    # 5. Save aggregate results
    all_events_combined = pd.concat(all_event_results, ignore_index=True)
    all_concepts_combined = pd.concat(all_concept_results, ignore_index=True)
    
    all_events_combined.to_csv(os.path.join(output_dir, 'event_importance.csv'), index=False)
    all_concepts_combined.to_csv(os.path.join(output_dir, 'concept_importance.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - event_importance.csv       : Per-event importance scores")
    print(f"  - concept_importance.csv     : Per-concept importance scores")
    print(f"  - ablation_analysis_*.png    : Visualizations")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature ablation analysis for LLM classifier"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained classifier checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of patients to analyze"
    )
    parser.add_argument(
        "--medical_dict_path",
        type=str,
        default=None,
        help="Path to MedicalDictTranslation2.csv (default: src/resources/MedicalDictTranslation2.csv)"
    )
    
    args = parser.parse_args()
    
    main(
        config_filepath=args.config_filepath,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        medical_dict_path=args.medical_dict_path
    )


