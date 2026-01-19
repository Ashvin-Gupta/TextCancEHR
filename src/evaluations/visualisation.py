# src/evaluation/visualization.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import brier_score_loss

def plot_classification_performance(labels, probs, output_dir):
    """
    Plots PR Curve, ROC Curve, and Metric vs. Threshold curves.
    
    Args:
        labels: Array of ground truth labels (0 or 1)
        probs: Array of positive class probabilities (floats 0-1)
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', linestyle='--')  # Diagonal random guess
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # 3. Metrics vs. Threshold (The most useful one for you!)
    # Note: thresholds array is 1 element shorter than precision/recall
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add marker for 0.5 default
    plt.axvline(x=0.5, color='k', linestyle=':', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    plt.close()

    # 4. Calibration Curve
    # Use 10 bins for calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=10, strategy='uniform'
    )
    
    # Calculate Brier score (lower is better, perfect = 0)
    brier_score = brier_score_loss(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model Calibration', markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linestyle='--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve (Brier Score = {brier_score:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()
    
    print(f"Plots saved to: {output_dir}")