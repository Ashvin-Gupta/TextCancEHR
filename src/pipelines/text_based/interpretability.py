# src/pipelines/text_based/interpretability.py

"""
Interpretability module for LLM-based classifier analysis.

Provides tools to analyze the logistic regression classification head:
- Extract and analyze classifier weights
- Compute risk trajectories via the Logit Lens approach
- Visualize per-token risk contributions
- Identify potentially spurious correlations
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class WeightAnalysis:
    """Container for classifier weight analysis results."""
    weights: np.ndarray  # (num_labels, hidden_size)
    bias: np.ndarray  # (num_labels,)
    cancer_weights: np.ndarray  # (hidden_size,) - weights for cancer class
    control_weights: np.ndarray  # (hidden_size,) - weights for control class
    diff_weights: np.ndarray  # (hidden_size,) - cancer - control (log-odds direction)
    top_positive_dims: List[Tuple[int, float]]  # dims pushing toward cancer
    top_negative_dims: List[Tuple[int, float]]  # dims pushing toward control
    statistics: Dict[str, float]


@dataclass
class RiskTrajectory:
    """Container for per-position risk analysis."""
    positions: np.ndarray  # (seq_len,)
    risk_scores: np.ndarray  # (seq_len,) - log-odds at each position
    probabilities: np.ndarray  # (seq_len,) - P(cancer) at each position
    tokens: List[str]  # decoded tokens
    token_ids: np.ndarray  # (seq_len,)
    top_risk_positions: List[Tuple[int, float, str]]  # (pos, score, token)
    final_prediction: float  # final P(cancer)
    true_label: Optional[int]


def load_classifier_for_analysis(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[nn.Module, Any, Dict]:
    """
    Load a trained LLMClassifier for interpretability analysis.
    
    Args:
        config_path: Path to the YAML config file
        checkpoint_path: Path to the trained classifier checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    import yaml
    from src.training.classification_trainer import LLMClassifier
    from src.training.utils import load_LoRA_model
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print(f"\n{'='*80}")
    print("Loading model for interpretability analysis")
    print(f"{'='*80}")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Load base model with LoRA adapters
    base_model, tokenizer = load_LoRA_model(config)
    
    # Create classifier wrapper
    classifier_model = LLMClassifier(
        base_model=base_model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=True,
        tokenizer=tokenizer
    )
    
    # Load trained classifier weights
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    safe_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights in {checkpoint_path}")
    
    classifier_model.load_state_dict(state_dict, strict=False)
    classifier_model.to(device)
    classifier_model.eval()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  ✓ Classifier head shape: {classifier_model.classifier.weight.shape}")
    
    return classifier_model, tokenizer, config


def extract_lr_weights(
    model: nn.Module,
    top_k: int = 50
) -> WeightAnalysis:
    """
    Extract and analyze the logistic regression classifier weights.
    
    For binary classification with CrossEntropyLoss:
    - W has shape (2, hidden_size)
    - W[1] - W[0] gives the log-odds direction (positive = more cancer-like)
    
    Args:
        model: LLMClassifier model
        top_k: Number of top dimensions to return
        
    Returns:
        WeightAnalysis containing weights and statistics
    """
    # Extract weights
    W = model.classifier.weight.detach().cpu().numpy()  # (num_labels, hidden_size)
    b = model.classifier.bias.detach().cpu().numpy()  # (num_labels,)
    
    # For binary classification: class 0 = control, class 1 = cancer
    control_weights = W[0]
    cancer_weights = W[1]
    
    # The difference W[1] - W[0] gives the log-odds direction
    # Positive values indicate dimensions that push toward cancer prediction
    diff_weights = cancer_weights - control_weights
    
    # Get top positive and negative dimensions
    sorted_indices = np.argsort(diff_weights)
    top_negative_indices = sorted_indices[:top_k]  # Most negative (push toward control)
    top_positive_indices = sorted_indices[-top_k:][::-1]  # Most positive (push toward cancer)
    
    top_positive_dims = [(int(idx), float(diff_weights[idx])) for idx in top_positive_indices]
    top_negative_dims = [(int(idx), float(diff_weights[idx])) for idx in top_negative_indices]
    
    # Compute statistics
    statistics = {
        "mean": float(np.mean(diff_weights)),
        "std": float(np.std(diff_weights)),
        "min": float(np.min(diff_weights)),
        "max": float(np.max(diff_weights)),
        "median": float(np.median(diff_weights)),
        "abs_mean": float(np.mean(np.abs(diff_weights))),
        "sparsity_1e-3": float(np.mean(np.abs(diff_weights) < 1e-3)),
        "sparsity_1e-2": float(np.mean(np.abs(diff_weights) < 1e-2)),
        "bias_diff": float(b[1] - b[0]),  # Baseline log-odds
        "cancer_bias": float(b[1]),
        "control_bias": float(b[0]),
    }
    
    return WeightAnalysis(
        weights=W,
        bias=b,
        cancer_weights=cancer_weights,
        control_weights=control_weights,
        diff_weights=diff_weights,
        top_positive_dims=top_positive_dims,
        top_negative_dims=top_negative_dims,
        statistics=statistics
    )


def compute_risk_trajectory(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    weight_analysis: WeightAnalysis,
    true_label: Optional[int] = None,
    max_length: Optional[int] = None,
    device: str = "cuda"
) -> RiskTrajectory:
    """
    Compute the Logit Lens risk trajectory for a patient sequence.
    
    Applies the classifier weights to hidden states at each position to see
    how the "risk score" evolves across the patient's timeline.
    
    Args:
        model: LLMClassifier model
        tokenizer: Tokenizer
        text: Patient text sequence
        weight_analysis: Pre-computed weight analysis
        true_label: Ground truth label (optional)
        max_length: Maximum sequence length
        device: Device to run on
        
    Returns:
        RiskTrajectory containing per-position risk scores
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    seq_len = input_ids.shape[1]
    
    # Get hidden states at all positions
    with torch.no_grad():
        backbone = getattr(model.base_model, "model", model.base_model)
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last layer hidden states: (1, seq_len, hidden_size)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
    
    # Move to CPU for analysis
    hidden_states = hidden_states[0].cpu().numpy()  # (seq_len, hidden_size)
    
    # Apply classifier weights to get log-odds at each position
    # logits[t] = W @ h[t] + b
    W = weight_analysis.weights  # (2, hidden_size)
    b = weight_analysis.bias  # (2,)
    
    # Compute logits at each position
    logits = hidden_states @ W.T + b  # (seq_len, 2)
    
    # Convert to log-odds (cancer vs control) and probabilities
    log_odds = logits[:, 1] - logits[:, 0]  # (seq_len,)
    probabilities = 1 / (1 + np.exp(-log_odds))  # Sigmoid
    
    # Decode tokens
    token_ids = input_ids[0].cpu().numpy()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Find top risk positions (highest absolute log-odds change)
    # We look at positions where risk is highest
    sorted_indices = np.argsort(log_odds)[::-1]
    top_risk_positions = [
        (int(idx), float(log_odds[idx]), tokens[idx])
        for idx in sorted_indices[:20]
    ]
    
    return RiskTrajectory(
        positions=np.arange(seq_len),
        risk_scores=log_odds,
        probabilities=probabilities,
        tokens=tokens,
        token_ids=token_ids,
        top_risk_positions=top_risk_positions,
        final_prediction=float(probabilities[-1]),
        true_label=true_label
    )


def get_top_contributing_tokens(
    trajectory: RiskTrajectory,
    top_k: int = 20,
    context_window: int = 5
) -> List[Dict[str, Any]]:
    """
    Find tokens at positions where risk score is highest.
    
    Args:
        trajectory: Computed risk trajectory
        top_k: Number of top positions to return
        context_window: Number of tokens before/after to include
        
    Returns:
        List of dictionaries with token info and context
    """
    results = []
    
    # Sort positions by risk score
    sorted_indices = np.argsort(trajectory.risk_scores)[::-1]
    
    for idx in sorted_indices[:top_k]:
        idx = int(idx)
        
        # Get context window
        start = max(0, idx - context_window)
        end = min(len(trajectory.tokens), idx + context_window + 1)
        
        context_tokens = trajectory.tokens[start:end]
        context_text = "".join(context_tokens)
        
        # Mark the target token
        relative_pos = idx - start
        
        results.append({
            "position": idx,
            "token": trajectory.tokens[idx],
            "token_id": int(trajectory.token_ids[idx]),
            "risk_score": float(trajectory.risk_scores[idx]),
            "probability": float(trajectory.probabilities[idx]),
            "context": context_text,
            "context_tokens": context_tokens,
            "target_in_context": relative_pos
        })
    
    return results


def visualize_weight_distribution(
    weight_analysis: WeightAnalysis,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Visualize the classifier weight distribution.
    
    Creates:
    - Histogram of weight values
    - Top positive/negative dimensions bar chart
    """
    os.makedirs(output_dir, exist_ok=True)
    
    diff_weights = weight_analysis.diff_weights
    stats = weight_analysis.statistics
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Histogram of weight differences
    ax1 = axes[0, 0]
    ax1.hist(diff_weights, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(x=stats['mean'], color='green', linestyle='--', linewidth=2, label=f"Mean={stats['mean']:.4f}")
    ax1.set_xlabel('Weight (Cancer - Control)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Log-Odds Weights', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top positive dimensions (push toward cancer)
    ax2 = axes[0, 1]
    top_pos = weight_analysis.top_positive_dims[:20]
    dims = [f"Dim {d[0]}" for d in top_pos]
    vals = [d[1] for d in top_pos]
    colors = ['crimson' if v > 0 else 'steelblue' for v in vals]
    bars = ax2.barh(dims[::-1], vals[::-1], color=colors[::-1], edgecolor='black')
    ax2.set_xlabel('Weight Value', fontsize=12)
    ax2.set_title('Top 20 Dimensions → Cancer', fontsize=14)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Top negative dimensions (push toward control)
    ax3 = axes[1, 0]
    top_neg = weight_analysis.top_negative_dims[:20]
    dims = [f"Dim {d[0]}" for d in top_neg]
    vals = [d[1] for d in top_neg]
    colors = ['steelblue' if v < 0 else 'crimson' for v in vals]
    bars = ax3.barh(dims[::-1], vals[::-1], color=colors[::-1], edgecolor='black')
    ax3.set_xlabel('Weight Value', fontsize=12)
    ax3.set_title('Top 20 Dimensions → Control', fontsize=14)
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Weight Statistics (Cancer - Control Direction)
    {'='*50}
    
    Mean:           {stats['mean']:.6f}
    Std:            {stats['std']:.6f}
    Min:            {stats['min']:.6f}
    Max:            {stats['max']:.6f}
    Median:         {stats['median']:.6f}
    
    Absolute Mean:  {stats['abs_mean']:.6f}
    
    Sparsity (<1e-3): {stats['sparsity_1e-3']*100:.1f}%
    Sparsity (<1e-2): {stats['sparsity_1e-2']*100:.1f}%
    
    Bias (Cancer):  {stats['cancer_bias']:.6f}
    Bias (Control): {stats['control_bias']:.6f}
    Bias Diff:      {stats['bias_diff']:.6f}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_distribution.png'), dpi=150)
    plt.close()
    
    print(f"  ✓ Weight distribution plot saved to {output_dir}")


def visualize_risk_trajectory(
    trajectory: RiskTrajectory,
    output_path: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    show_tokens: bool = False,
    subsample_factor: int = 1
) -> None:
    """
    Visualize the risk trajectory across a patient sequence.
    
    Args:
        trajectory: Computed risk trajectory
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        show_tokens: Whether to show token labels (only for short sequences)
        subsample_factor: Subsample positions for cleaner visualization
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    positions = trajectory.positions[::subsample_factor]
    risk_scores = trajectory.risk_scores[::subsample_factor]
    probabilities = trajectory.probabilities[::subsample_factor]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Risk scores (log-odds)
    ax1.plot(positions, risk_scores, color='crimson', linewidth=1, alpha=0.8)
    ax1.fill_between(positions, 0, risk_scores, 
                     where=(risk_scores > 0), color='crimson', alpha=0.3, label='Cancer risk')
    ax1.fill_between(positions, 0, risk_scores,
                     where=(risk_scores <= 0), color='steelblue', alpha=0.3, label='Control risk')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Log-Odds (Cancer vs Control)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if title:
        ax1.set_title(title, fontsize=14)
    
    # Plot 2: Probability
    ax2.plot(positions, probabilities, color='purple', linewidth=1, alpha=0.8)
    ax2.fill_between(positions, 0.5, probabilities,
                     where=(probabilities > 0.5), color='crimson', alpha=0.3)
    ax2.fill_between(positions, 0.5, probabilities,
                     where=(probabilities <= 0.5), color='steelblue', alpha=0.3)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('P(Cancer)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add label information
    label_str = "Unknown"
    if trajectory.true_label is not None:
        label_str = "Cancer" if trajectory.true_label > 0 else "Control"
    
    info_text = f"True Label: {label_str}\nFinal P(Cancer): {trajectory.final_prediction:.3f}"
    ax2.text(0.98, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_top_risk_tokens(
    trajectory: RiskTrajectory,
    output_path: str,
    top_k: int = 30,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Visualize the tokens with highest risk scores.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get top and bottom risk positions
    sorted_indices = np.argsort(trajectory.risk_scores)
    
    top_cancer_idx = sorted_indices[-top_k:][::-1]
    top_control_idx = sorted_indices[:top_k]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Top cancer tokens
    tokens = [f"[{i}] {trajectory.tokens[i][:15]}" for i in top_cancer_idx]
    scores = [trajectory.risk_scores[i] for i in top_cancer_idx]
    ax1.barh(tokens[::-1], scores[::-1], color='crimson', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Log-Odds Score', fontsize=12)
    ax1.set_title(f'Top {top_k} Tokens → Cancer', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Top control tokens
    tokens = [f"[{i}] {trajectory.tokens[i][:15]}" for i in top_control_idx]
    scores = [trajectory.risk_scores[i] for i in top_control_idx]
    ax2.barh(tokens[::-1], scores[::-1], color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Log-Odds Score', fontsize=12)
    ax2.set_title(f'Top {top_k} Tokens → Control', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_analysis_results(
    weight_analysis: WeightAnalysis,
    output_dir: str
) -> None:
    """
    Save weight analysis results to JSON and numpy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save weights as numpy
    np.save(os.path.join(output_dir, 'classifier_weights.npy'), weight_analysis.weights)
    np.save(os.path.join(output_dir, 'classifier_bias.npy'), weight_analysis.bias)
    np.save(os.path.join(output_dir, 'diff_weights.npy'), weight_analysis.diff_weights)
    
    # Save analysis as JSON
    analysis_dict = {
        "statistics": weight_analysis.statistics,
        "top_positive_dims": weight_analysis.top_positive_dims,
        "top_negative_dims": weight_analysis.top_negative_dims,
    }
    
    with open(os.path.join(output_dir, 'weight_analysis.json'), 'w') as f:
        json.dump(analysis_dict, f, indent=2)
    
    print(f"  ✓ Analysis results saved to {output_dir}")


def detect_spurious_patterns(
    trajectories: List[RiskTrajectory],
    tokenizer: Any,
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Analyze multiple trajectories to detect potentially spurious patterns.
    
    Looks for:
    - Non-clinical tokens that frequently appear in high-risk positions
    - Tokens that are highly predictive but seem unrelated to medical content
    
    Args:
        trajectories: List of risk trajectories
        tokenizer: Tokenizer for decoding
        top_k: Number of top tokens to analyze
        
    Returns:
        Dictionary with spurious pattern analysis
    """
    from collections import Counter
    
    # Collect tokens at high-risk positions across all trajectories
    high_risk_tokens = Counter()
    low_risk_tokens = Counter()
    
    for traj in trajectories:
        # Get top 10% risk positions
        threshold_high = np.percentile(traj.risk_scores, 90)
        threshold_low = np.percentile(traj.risk_scores, 10)
        
        for i, (token, score) in enumerate(zip(traj.tokens, traj.risk_scores)):
            if score >= threshold_high:
                high_risk_tokens[token] += 1
            elif score <= threshold_low:
                low_risk_tokens[token] += 1
    
    # Suspicious patterns to check
    suspicious_patterns = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "dr", "dr.", "clinic", "ward", "room", "bed",
        "am", "pm", ":", "/", "-",
    ]
    
    # Check for suspicious tokens in high-risk list
    warnings = []
    high_risk_list = high_risk_tokens.most_common(top_k)
    
    for token, count in high_risk_list:
        token_lower = token.lower().strip()
        for pattern in suspicious_patterns:
            if pattern in token_lower:
                warnings.append({
                    "token": token,
                    "count": count,
                    "pattern_match": pattern,
                    "warning": f"Non-clinical token '{token}' appears {count} times in high-risk positions"
                })
                break
    
    return {
        "high_risk_tokens": high_risk_list,
        "low_risk_tokens": low_risk_tokens.most_common(top_k),
        "warnings": warnings,
        "num_warnings": len(warnings)
    }

