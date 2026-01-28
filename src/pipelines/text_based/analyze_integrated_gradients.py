# src/pipelines/text_based/analyze_integrated_gradients.py

"""
Analysis script for running Integrated Gradients on LLM classifier.

Computes IG attributions for sample patients, aggregates to event and concept levels,
and creates comprehensive visualizations.

Usage:
    python -m src.pipelines.text_based.analyze_integrated_gradients \
        --config_filepath path/to/config.yaml \
        --checkpoint_path path/to/checkpoint \
        --output_dir ./ig_results \
        --num_samples 10 \
        --n_steps 50 \
        --baseline_strategy pad
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import List, Dict
import json

from src.pipelines.text_based.integrated_gradients import (
    compute_integrated_gradients,
    aggregate_attributions_to_events,
    aggregate_attributions_to_concepts,
    visualize_token_attributions,
    TokenAttributions,
)
from src.pipelines.text_based.interpretability import load_classifier_for_analysis
from src.pipelines.text_based.feature_ablation_analysis import load_medical_terms
from src.data.unified_dataset import UnifiedEHRDataset
from src.training.utils import seed_all


def print_ig_summary(token_attrs: TokenAttributions, patient_idx: int) -> None:
    """Print a formatted summary of IG results."""
    print(f"\n📋 Patient {patient_idx} - Integrated Gradients Results")
    print("-" * 70)
    print(f"  Sequence Length:        {len(token_attrs.tokens)} tokens")
    print(f"  Baseline P(Cancer):     {token_attrs.baseline_prediction:.4f}")
    print(f"  Input P(Cancer):        {token_attrs.input_prediction:.4f}")
    print(f"  Prediction Change:      {token_attrs.input_prediction - token_attrs.baseline_prediction:+.4f}")
    print(f"  Completeness Score:     {token_attrs.completeness_score:.4f} (lower is better)")
    
    # Attribution statistics
    print(f"\n  Attribution Statistics:")
    print(f"    Mean:                 {np.mean(token_attrs.attributions):.6f}")
    print(f"    Std:                  {np.std(token_attrs.attributions):.6f}")
    print(f"    Min:                  {np.min(token_attrs.attributions):.6f}")
    print(f"    Max:                  {np.max(token_attrs.attributions):.6f}")
    print(f"    Sum:                  {np.sum(token_attrs.attributions):.6f}")
    
    # Top attributions
    print(f"\n  🔝 Top 10 Positive Attributions (→ Cancer):")
    sorted_indices = np.argsort(token_attrs.attributions)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        token = token_attrs.tokens[idx].replace('\n', '\\n')[:30]
        print(f"      Position {idx:5d}: {token_attrs.attributions[idx]:+.4f}  '{token}'")
    
    print(f"\n  🔻 Top 10 Negative Attributions (→ Control):")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        token = token_attrs.tokens[idx].replace('\n', '\\n')[:30]
        print(f"      Position {idx:5d}: {token_attrs.attributions[idx]:+.4f}  '{token}'")


def visualize_event_attributions(
    event_attrs: List[Dict],
    output_path: str,
    top_k: int = 20
) -> None:
    """Visualize event-level attributions across multiple patients."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Aggregate across patients
    all_events = []
    for patient_events in event_attrs:
        all_events.extend(patient_events['events'])
    
    # Sort by absolute attribution
    all_events.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top positive and negative
    positive_events = [e for e in all_events if e[1] > 0][:top_k]
    negative_events = [e for e in all_events if e[1] < 0][:top_k]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Top positive events
    if positive_events:
        event_texts = [e[0][:40] + '...' if len(e[0]) > 40 else e[0] for e in positive_events]
        event_scores = [e[1] for e in positive_events]
        
        ax1.barh(range(len(event_texts)), event_scores, color='crimson', edgecolor='black', alpha=0.7)
        ax1.set_yticks(range(len(event_texts)))
        ax1.set_yticklabels(event_texts, fontsize=8)
        ax1.set_xlabel('Attribution Score', fontsize=12)
        ax1.set_title(f'Top {top_k} Events → Cancer (Positive Attribution)', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
    
    # Top negative events
    if negative_events:
        event_texts = [e[0][:40] + '...' if len(e[0]) > 40 else e[0] for e in negative_events]
        event_scores = [e[1] for e in negative_events]
        
        ax2.barh(range(len(event_texts)), event_scores, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_yticks(range(len(event_texts)))
        ax2.set_yticklabels(event_texts, fontsize=8)
        ax2.set_xlabel('Attribution Score', fontsize=12)
        ax2.set_title(f'Top {top_k} Events → Control (Negative Attribution)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_concept_attributions(
    concept_attrs: List[Dict],
    output_path: str,
    top_k: int = 25
) -> None:
    """Visualize medical concept attributions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Aggregate concepts across patients
    concept_scores = {}
    concept_categories = {}
    
    for patient_concepts in concept_attrs:
        for concept_text, concept_code, score, category in patient_concepts['concepts']:
            if concept_code not in concept_scores:
                concept_scores[concept_code] = []
                concept_categories[concept_code] = category
            concept_scores[concept_code].append(score)
    
    # Average scores per concept
    concept_avg = {code: np.mean(scores) for code, scores in concept_scores.items()}
    
    # Sort by absolute attribution
    sorted_concepts = sorted(concept_avg.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Get top positive and negative
    positive_concepts = [(c, s) for c, s in sorted_concepts if s > 0][:top_k]
    negative_concepts = [(c, s) for c, s in sorted_concepts if s < 0][:top_k]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top positive concepts
    ax1 = axes[0, 0]
    if positive_concepts:
        concept_names = [c[:30] for c, _ in positive_concepts]
        concept_scores_pos = [s for _, s in positive_concepts]
        
        ax1.barh(range(len(concept_names)), concept_scores_pos, color='crimson', edgecolor='black', alpha=0.7)
        ax1.set_yticks(range(len(concept_names)))
        ax1.set_yticklabels(concept_names, fontsize=8)
        ax1.set_xlabel('Average Attribution', fontsize=11)
        ax1.set_title(f'Top {top_k} Concepts → Cancer', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
    
    # Top negative concepts
    ax2 = axes[0, 1]
    if negative_concepts:
        concept_names = [c[:30] for c, _ in negative_concepts]
        concept_scores_neg = [s for _, s in negative_concepts]
        
        ax2.barh(range(len(concept_names)), concept_scores_neg, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_yticks(range(len(concept_names)))
        ax2.set_yticklabels(concept_names, fontsize=8)
        ax2.set_xlabel('Average Attribution', fontsize=11)
        ax2.set_title(f'Top {top_k} Concepts → Control', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
    
    # Category analysis
    ax3 = axes[1, 0]
    category_scores = {}
    for code, scores in concept_scores.items():
        category = concept_categories[code]
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].extend(scores)
    
    category_avg = {cat: np.mean(scores) for cat, scores in category_scores.items()}
    sorted_categories = sorted(category_avg.items(), key=lambda x: x[1], reverse=True)
    
    cat_names = [c for c, _ in sorted_categories]
    cat_scores = [s for _, s in sorted_categories]
    
    bars = ax3.bar(range(len(cat_names)), cat_scores, color='purple', edgecolor='black', alpha=0.7)
    ax3.set_xticks(range(len(cat_names)))
    ax3.set_xticklabels(cat_names, rotation=45, ha='right')
    ax3.set_ylabel('Average Attribution', fontsize=11)
    ax3.set_title('Average Attribution by Concept Category', fontsize=12)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Distribution
    ax4 = axes[1, 1]
    all_scores = [s for scores in concept_scores.values() for s in scores]
    ax4.hist(all_scores, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Attribution Score', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Concept Attributions', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(
    config_filepath: str,
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 10,
    n_steps: int = 50,
    baseline_strategy: str = 'pad',
    medical_dict_path: str = None
):
    """Main IG analysis pipeline."""
    seed_all(42)
    
    print("\n" + "=" * 80)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("=" * 80)
    print(f"\nConfig:     {config_filepath}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_dir}")
    print(f"N Steps:    {n_steps}")
    print(f"Baseline:   {baseline_strategy}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    
    # Check if Captum is installed
    try:
        import captum
        print(f"Captum:     v{captum.__version__}")
    except ImportError:
        print("\n⚠️  ERROR: Captum library not found!")
        print("Install with: pip install captum")
        return
    
    # Load model
    model, tokenizer, config = load_classifier_for_analysis(
        config_path=config_filepath,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Load medical terms
    if medical_dict_path is None:
        medical_dict_path = "src/resources/MedicalDictTranslation2.csv"
    
    print(f"\nLoading medical dictionary from: {medical_dict_path}")
    medical_terms = load_medical_terms(medical_dict_path, max_terms=382)
    print(f"  Loaded {len(medical_terms)} terms of interest")
    
    # Load dataset
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
    
    # Find cancer patients
    cancer_indices = []
    for i in range(min(len(val_dataset), 500)):
        sample = val_dataset[i]
        if sample is not None:
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            if label > 0:
                cancer_indices.append(i)
    
    print(f"  Found {len(cancer_indices)} cancer patients")
    
    # Sample patients
    sample_patients = random.sample(cancer_indices, min(num_samples, len(cancer_indices)))
    
    # Storage for results
    all_token_results = []
    all_event_results = []
    all_concept_results = []
    
    print("\n" + "=" * 80)
    print("COMPUTING INTEGRATED GRADIENTS")
    print("=" * 80)
    
    for patient_num, idx in enumerate(sample_patients, 1):
        sample = val_dataset[idx]
        text = sample['text']
        label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
        
        print(f"\n{'='*70}")
        print(f"Patient {idx} ({patient_num}/{len(sample_patients)}) - Label: {'Cancer' if label > 0 else 'Control'}")
        print(f"{'='*70}")
        print(f"Text length: {len(text)} characters")
        
        # Compute IG
        print(f"Computing Integrated Gradients ({n_steps} steps)...")
        token_attrs = compute_integrated_gradients(
            model=model,
            tokenizer=tokenizer,
            text=text,
            baseline_strategy=baseline_strategy,
            n_steps=n_steps,
            max_length=12000,
            device=device
        )
        
        # Print summary
        print_ig_summary(token_attrs, idx)
        
        # Save token attributions
        np.save(
            os.path.join(output_dir, f'ig_attributions_patient_{idx}.npy'),
            token_attrs.attributions
        )
        
        # Visualize tokens
        token_viz_path = os.path.join(output_dir, f'ig_tokens_patient_{idx}.png')
        visualize_token_attributions(token_attrs, token_viz_path)
        
        # Aggregate to events
        event_attrs = aggregate_attributions_to_events(token_attrs, delimiter=';', aggregation_method='sum')
        
        print(f"\n  📊 Event Aggregation ({len(event_attrs.event_attributions)} events):")
        print(f"    Top 5 Events → Cancer:")
        sorted_events = sorted(event_attrs.event_attributions, key=lambda x: x[1], reverse=True)
        for i, (event_text, score) in enumerate(sorted_events[:5], 1):
            print(f"      {i}. {score:+.4f}: {event_text[:60]}...")
        
        all_event_results.append({
            'patient_id': idx,
            'events': event_attrs.event_attributions
        })
        
        # Aggregate to concepts
        concept_attrs = aggregate_attributions_to_concepts(token_attrs, medical_terms, aggregation_method='mean')
        
        print(f"\n  🏥 Concept Aggregation ({len(concept_attrs.concept_attributions)} concepts):")
        print(f"    Top 5 Concepts → Cancer:")
        sorted_concepts = sorted(concept_attrs.concept_attributions, key=lambda x: x[2], reverse=True)
        for i, (concept_text, concept_code, score) in enumerate(sorted_concepts[:5], 1):
            print(f"      {i}. {score:+.4f}: {concept_text} ({concept_code})")
        
        all_concept_results.append({
            'patient_id': idx,
            'concepts': [(c[0], c[1], c[2], cat) for c, cat in zip(concept_attrs.concept_attributions, concept_attrs.concept_categories)]
        })
        
        all_token_results.append({
            'patient_id': idx,
            'label': label,
            'baseline_prob': token_attrs.baseline_prediction,
            'input_prob': token_attrs.input_prediction,
            'completeness_score': token_attrs.completeness_score,
            'attribution_mean': float(np.mean(token_attrs.attributions)),
            'attribution_std': float(np.std(token_attrs.attributions)),
        })
    
    # Create aggregate visualizations
    print("\n" + "=" * 80)
    print("CREATING AGGREGATE VISUALIZATIONS")
    print("=" * 80)
    
    visualize_event_attributions(
        all_event_results,
        os.path.join(output_dir, 'ig_events_aggregate.png')
    )
    
    visualize_concept_attributions(
        all_concept_results,
        os.path.join(output_dir, 'ig_concepts_aggregate.png')
    )
    
    # Save results to CSV
    token_results_df = pd.DataFrame(all_token_results)
    token_results_df.to_csv(os.path.join(output_dir, 'ig_summary.csv'), index=False)
    
    # Save metadata
    metadata = {
        'n_steps': n_steps,
        'baseline_strategy': baseline_strategy,
        'num_patients': len(sample_patients),
        'patient_ids': [int(i) for i in sample_patients],
        'config_path': config_filepath,
        'checkpoint_path': checkpoint_path,
    }
    
    with open(os.path.join(output_dir, 'ig_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - ig_summary.csv                    : Per-patient summary statistics")
    print(f"  - ig_attributions_patient_*.npy     : Raw attribution arrays")
    print(f"  - ig_tokens_patient_*.png           : Token-level visualizations")
    print(f"  - ig_events_aggregate.png           : Aggregate event attributions")
    print(f"  - ig_concepts_aggregate.png         : Aggregate concept attributions")
    print(f"  - ig_metadata.json                  : Analysis parameters")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated Gradients analysis for LLM classifier"
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
        default="./ig_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of patients to analyze"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of steps for IG approximation (more = more accurate but slower)"
    )
    parser.add_argument(
        "--baseline_strategy",
        type=str,
        default="pad",
        choices=["pad", "unk", "mask", "zero"],
        help="Baseline strategy for IG"
    )
    parser.add_argument(
        "--medical_dict_path",
        type=str,
        default=None,
        help="Path to MedicalDictTranslation2.csv"
    )
    
    args = parser.parse_args()
    
    main(
        config_filepath=args.config_filepath,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        baseline_strategy=args.baseline_strategy,
        medical_dict_path=args.medical_dict_path
    )

