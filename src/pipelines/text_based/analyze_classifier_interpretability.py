# src/pipelines/text_based/analyze_classifier_interpretability.py

"""
Standalone script for analyzing LLM classifier interpretability.

Loads a trained classifier checkpoint and performs:
1. Logistic Regression coefficient analysis
2. Risk trajectory visualization (Logit Lens)
3. Spurious correlation detection

Usage:
    python -m src.pipelines.text_based.analyze_classifier_interpretability \
        --config_filepath path/to/config.yaml \
        --checkpoint_path path/to/checkpoint \
        --output_dir path/to/output

Example:
    python -m src.pipelines.text_based.analyze_classifier_interpretability \
        --config_filepath src/pipelines/text_based/configs/llm_classify_pretrained_cls_lora.yaml \
        --checkpoint_path /data/scratch/qc25022/pancreas/experiments/lora-6-month-logistic-raw/checkpoint-7856 \
        --output_dir ./interpretability_results
"""

import argparse
import os
import yaml
import torch
import numpy as np
import random
from typing import List, Tuple

from src.pipelines.text_based.interpretability import (
    load_classifier_for_analysis,
    extract_lr_weights,
    compute_risk_trajectory,
    get_top_contributing_tokens,
    visualize_weight_distribution,
    visualize_risk_trajectory,
    visualize_top_risk_tokens,
    save_analysis_results,
    detect_spurious_patterns,
    WeightAnalysis,
    RiskTrajectory,
)
from src.data.unified_dataset import UnifiedEHRDataset
from src.training.utils import seed_all


def print_weight_analysis_summary(weight_analysis: WeightAnalysis) -> None:
    """Print a formatted summary of weight analysis."""
    stats = weight_analysis.statistics
    
    print("\n" + "=" * 80)
    print("CLASSIFIER WEIGHT ANALYSIS (Logistic Regression Coefficients)")
    print("=" * 80)
    
    print("\n📊 Weight Statistics (Cancer - Control direction):")
    print("-" * 50)
    print(f"  Mean:           {stats['mean']:.6f}")
    print(f"  Std:            {stats['std']:.6f}")
    print(f"  Min:            {stats['min']:.6f}")
    print(f"  Max:            {stats['max']:.6f}")
    print(f"  Median:         {stats['median']:.6f}")
    print(f"  Absolute Mean:  {stats['abs_mean']:.6f}")
    
    print("\n📈 Sparsity:")
    print("-" * 50)
    print(f"  Weights < 1e-3:  {stats['sparsity_1e-3']*100:.1f}%")
    print(f"  Weights < 1e-2:  {stats['sparsity_1e-2']*100:.1f}%")
    
    print("\n⚖️  Bias Terms:")
    print("-" * 50)
    print(f"  Cancer Bias:    {stats['cancer_bias']:.6f}")
    print(f"  Control Bias:   {stats['control_bias']:.6f}")
    print(f"  Bias Diff:      {stats['bias_diff']:.6f} (baseline log-odds)")
    
    print("\n🔺 Top 20 Dimensions → CANCER (Highest Positive Weights):")
    print("-" * 50)
    for i, (dim, weight) in enumerate(weight_analysis.top_positive_dims[:20], 1):
        print(f"  {i:2d}. Dimension {dim:4d}: {weight:+.6f}")
    
    print("\n🔻 Top 20 Dimensions → CONTROL (Highest Negative Weights):")
    print("-" * 50)
    for i, (dim, weight) in enumerate(weight_analysis.top_negative_dims[:20], 1):
        print(f"  {i:2d}. Dimension {dim:4d}: {weight:+.6f}")
    
    print("=" * 80 + "\n")


def print_trajectory_summary(trajectory: RiskTrajectory, patient_idx: int) -> None:
    """Print a formatted summary of a risk trajectory."""
    label_str = "Cancer" if trajectory.true_label > 0 else "Control"
    
    print(f"\n📋 Patient {patient_idx} - True Label: {label_str}")
    print("-" * 60)
    print(f"  Sequence Length:      {len(trajectory.tokens)} tokens")
    print(f"  Final P(Cancer):      {trajectory.final_prediction:.4f}")
    print(f"  Final Log-Odds:       {trajectory.risk_scores[-1]:.4f}")
    print(f"  Max Risk Score:       {np.max(trajectory.risk_scores):.4f}")
    print(f"  Min Risk Score:       {np.min(trajectory.risk_scores):.4f}")
    print(f"  Mean Risk Score:      {np.mean(trajectory.risk_scores):.4f}")
    
    print(f"\n  🔝 Top 10 High-Risk Positions:")
    for pos, score, token in trajectory.top_risk_positions[:10]:
        token_display = token.replace('\n', '\\n')[:20]
        print(f"      Position {pos:5d}: {score:+.4f}  '{token_display}'")


def analyze_patient_samples(
    model: torch.nn.Module,
    tokenizer,
    dataset: UnifiedEHRDataset,
    weight_analysis: WeightAnalysis,
    output_dir: str,
    num_samples: int = 10,
    device: str = "cuda"
) -> Tuple[List[RiskTrajectory], List[RiskTrajectory]]:
    """
    Analyze risk trajectories for sample patients.
    
    Args:
        model: Loaded classifier model
        tokenizer: Tokenizer
        dataset: Dataset to sample from
        weight_analysis: Pre-computed weight analysis
        output_dir: Directory to save trajectory plots
        num_samples: Number of samples per class
        device: Device to run on
        
    Returns:
        Tuple of (cancer_trajectories, control_trajectories)
    """
    print("\n" + "=" * 80)
    print("RISK TRAJECTORY ANALYSIS (Logit Lens)")
    print("=" * 80)
    
    # Separate cancer and control indices
    cancer_indices = []
    control_indices = []
    
    print("\nScanning dataset for cancer/control patients...")
    for i in range(min(len(dataset), 1000)):  # Scan first 1000
        sample = dataset[i]
        if sample is not None:
            label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
            if label > 0:
                cancer_indices.append(i)
            else:
                control_indices.append(i)
    
    print(f"  Found {len(cancer_indices)} cancer patients")
    print(f"  Found {len(control_indices)} control patients")
    if cancer_indices:
        sample_cancer_labels = [dataset[i]['label'].item() if torch.is_tensor(dataset[i]['label']) else dataset[i]['label'] 
                            for i in cancer_indices[:5]]
        print(f"  Sample cancer labels: {sample_cancer_labels}")
    
    # Sample from each class
    cancer_sample = random.sample(cancer_indices, min(num_samples, len(cancer_indices)))
    control_sample = random.sample(control_indices, min(num_samples, len(control_indices)))
    
    cancer_trajectories = []
    control_trajectories = []
    
    trajectory_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    # Analyze cancer patients
    print(f"\n🔴 Analyzing {len(cancer_sample)} CANCER patients:")
    for i, idx in enumerate(cancer_sample):
        sample = dataset[idx]
        text = sample['text']
        label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
        
        trajectory = compute_risk_trajectory(
            model=model,
            tokenizer=tokenizer,
            text=text,
            weight_analysis=weight_analysis,
            true_label=label,
            max_length=12000,
            device=device
        )
        
        cancer_trajectories.append(trajectory)
        print_trajectory_summary(trajectory, idx)
        
        # Save trajectory plot
        plot_path = os.path.join(trajectory_dir, f"cancer_patient_{idx}.png")
        visualize_risk_trajectory(
            trajectory=trajectory,
            output_path=plot_path,
            title=f"Risk Trajectory - Cancer Patient {idx}",
            subsample_factor=max(1, len(trajectory.tokens) // 500)
        )
        
        # Save top risk tokens plot
        tokens_path = os.path.join(trajectory_dir, f"cancer_patient_{idx}_tokens.png")
        visualize_top_risk_tokens(
            trajectory=trajectory,
            output_path=tokens_path,
            top_k=25
        )
    
    # Analyze control patients
    print(f"\n🔵 Analyzing {len(control_sample)} CONTROL patients:")
    for i, idx in enumerate(control_sample):
        sample = dataset[idx]
        text = sample['text']
        label = sample['label'].item() if torch.is_tensor(sample['label']) else sample['label']
        
        trajectory = compute_risk_trajectory(
            model=model,
            tokenizer=tokenizer,
            text=text,
            weight_analysis=weight_analysis,
            true_label=label,
            max_length=12000,
            device=device
        )
        
        control_trajectories.append(trajectory)
        print_trajectory_summary(trajectory, idx)
        
        # Save trajectory plot
        plot_path = os.path.join(trajectory_dir, f"control_patient_{idx}.png")
        visualize_risk_trajectory(
            trajectory=trajectory,
            output_path=plot_path,
            title=f"Risk Trajectory - Control Patient {idx}",
            subsample_factor=max(1, len(trajectory.tokens) // 500)
        )
        
        # Save top risk tokens plot
        tokens_path = os.path.join(trajectory_dir, f"control_patient_{idx}_tokens.png")
        visualize_top_risk_tokens(
            trajectory=trajectory,
            output_path=tokens_path,
            top_k=25
        )
    
    print(f"\n  ✓ Trajectory plots saved to {trajectory_dir}")
    
    return cancer_trajectories, control_trajectories


def run_spurious_detection(
    cancer_trajectories: List[RiskTrajectory],
    control_trajectories: List[RiskTrajectory],
    tokenizer,
    output_dir: str
) -> None:
    """Run and report spurious pattern detection."""
    print("\n" + "=" * 80)
    print("SPURIOUS CORRELATION DETECTION")
    print("=" * 80)
    
    all_trajectories = cancer_trajectories + control_trajectories
    
    if len(all_trajectories) < 5:
        print("\n⚠️  Not enough trajectories for reliable spurious detection (need at least 5)")
        return
    
    results = detect_spurious_patterns(all_trajectories, tokenizer, top_k=50)
    
    print(f"\n🔍 Analysis of {len(all_trajectories)} patient trajectories:")
    
    print("\n📈 Most Common High-Risk Tokens:")
    print("-" * 50)
    for i, (token, count) in enumerate(results['high_risk_tokens'][:20], 1):
        token_display = token.replace('\n', '\\n')[:30]
        print(f"  {i:2d}. '{token_display}': {count} occurrences")
    
    print("\n📉 Most Common Low-Risk Tokens:")
    print("-" * 50)
    for i, (token, count) in enumerate(results['low_risk_tokens'][:20], 1):
        token_display = token.replace('\n', '\\n')[:30]
        print(f"  {i:2d}. '{token_display}': {count} occurrences")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({results['num_warnings']} potential spurious patterns):")
        print("-" * 50)
        for warning in results['warnings']:
            print(f"  - {warning['warning']}")
            print(f"    Pattern match: '{warning['pattern_match']}'")
    else:
        print("\n✅ No obvious spurious patterns detected in high-risk tokens")
    
    # Save results
    import json
    spurious_path = os.path.join(output_dir, "spurious_analysis.json")
    with open(spurious_path, 'w') as f:
        # Convert to serializable format
        serializable_results = {
            "high_risk_tokens": [(str(t), c) for t, c in results['high_risk_tokens']],
            "low_risk_tokens": [(str(t), c) for t, c in results['low_risk_tokens']],
            "warnings": results['warnings'],
            "num_warnings": results['num_warnings']
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n  ✓ Spurious analysis saved to {spurious_path}")


def main(config_filepath: str, checkpoint_path: str, output_dir: str, num_samples: int = 10):
    """Main analysis pipeline."""
    seed_all(42)
    
    print("\n" + "=" * 80)
    print("LLM CLASSIFIER INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nConfig:     {config_filepath}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device}")
    
    # 1. Load model
    model, tokenizer, config = load_classifier_for_analysis(
        config_path=config_filepath,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # 2. Extract and analyze weights
    print("\nExtracting classifier weights...")
    weight_analysis = extract_lr_weights(model, top_k=50)
    print_weight_analysis_summary(weight_analysis)
    
    # 3. Visualize weight distribution
    print("Generating weight distribution plots...")
    visualize_weight_distribution(weight_analysis, output_dir)
    
    # 4. Save weight analysis
    print("Saving weight analysis results...")
    save_analysis_results(weight_analysis, output_dir)
    
    # 5. Load dataset for trajectory analysis
    print("\nLoading dataset for trajectory analysis...")
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
    
    # Use validation set for analysis
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  Loaded {len(val_dataset)} validation samples")
    
    # 6. Analyze patient trajectories
    cancer_trajectories, control_trajectories = analyze_patient_samples(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
        weight_analysis=weight_analysis,
        output_dir=output_dir,
        num_samples=num_samples,
        device=device
    )
    
    # 7. Run spurious detection
    run_spurious_detection(
        cancer_trajectories=cancer_trajectories,
        control_trajectories=control_trajectories,
        tokenizer=tokenizer,
        output_dir=output_dir
    )
    
    # 8. Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - weight_distribution.png     : Classifier weight visualization")
    print(f"  - weight_analysis.json        : Weight statistics and top dimensions")
    print(f"  - classifier_weights.npy      : Raw classifier weights")
    print(f"  - diff_weights.npy            : Cancer-Control weight differences")
    print(f"  - trajectories/               : Per-patient risk trajectory plots")
    print(f"  - spurious_analysis.json      : Spurious correlation detection results")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze LLM classifier interpretability via LR coefficients and Logit Lens"
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
        default="./interpretability_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of patients to sample per class for trajectory analysis"
    )
    
    args = parser.parse_args()
    
    main(
        config_filepath=args.config_filepath,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

