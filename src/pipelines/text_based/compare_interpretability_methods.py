# src/pipelines/text_based/compare_interpretability_methods.py

"""
Compare three interpretability methods: LR Weights, Feature Ablation, and Integrated Gradients.

Analyzes agreement, correlation, and consensus across methods to identify
robust and reliable feature importance signals.

Usage:
    python -m src.pipelines.text_based.compare_interpretability_methods \
        --lr_dir ./interpretability_results \
        --ablation_dir ./ablation_results \
        --ig_dir ./ig_results \
        --output_dir ./method_comparison
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple
import json


def load_lr_results(lr_dir: str) -> Dict:
    """Load LR coefficient analysis results."""
    results = {}
    
    # Load weight analysis
    weights_path = os.path.join(lr_dir, 'weight_analysis.json')
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            results['weight_analysis'] = json.load(f)
    
    # Load diff weights
    diff_weights_path = os.path.join(lr_dir, 'diff_weights.npy')
    if os.path.exists(diff_weights_path):
        results['diff_weights'] = np.load(diff_weights_path)
    
    return results


def load_ablation_results(ablation_dir: str) -> Dict:
    """Load feature ablation results."""
    results = {}
    
    # Load event importance
    event_path = os.path.join(ablation_dir, 'event_importance.csv')
    if os.path.exists(event_path):
        results['events'] = pd.read_csv(event_path)
    
    # Load concept importance
    concept_path = os.path.join(ablation_dir, 'concept_importance.csv')
    if os.path.exists(concept_path):
        results['concepts'] = pd.read_csv(concept_path)
    
    return results


def load_ig_results(ig_dir: str) -> Dict:
    """Load Integrated Gradients results."""
    results = {}
    
    # Load summary
    summary_path = os.path.join(ig_dir, 'ig_summary.csv')
    if os.path.exists(summary_path):
        results['summary'] = pd.read_csv(summary_path)
    
    # Load metadata
    metadata_path = os.path.join(ig_dir, 'ig_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)
    
    return results


def compare_concept_importance(
    ablation_concepts: pd.DataFrame,
    output_dir: str
) -> pd.DataFrame:
    """
    Compare concept importance between ablation and IG methods.
    
    For now, focuses on ablation results. IG concept aggregation would need
    to be loaded separately if available.
    """
    # Aggregate by concept code
    concept_agg = ablation_concepts.groupby('concept_code').agg({
        'importance_score': ['mean', 'std', 'count'],
        'concept_category': 'first',
        'concept_text': 'first'
    }).reset_index()
    
    concept_agg.columns = ['concept_code', 'mean_importance', 'std_importance', 
                           'count', 'category', 'text']
    
    # Sort by absolute importance
    concept_agg['abs_importance'] = concept_agg['mean_importance'].abs()
    concept_agg = concept_agg.sort_values('abs_importance', ascending=False)
    
    return concept_agg


def visualize_method_comparison(
    lr_results: Dict,
    ablation_results: Dict,
    ig_results: Dict,
    output_dir: str
) -> None:
    """Create comprehensive comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. LR Weight Distribution
    if 'diff_weights' in lr_results:
        ax1 = fig.add_subplot(gs[0, 0])
        diff_weights = lr_results['diff_weights']
        ax1.hist(diff_weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Weight (Cancer - Control)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('LR Weights Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # 2. Ablation Event Importance Distribution
    if 'events' in ablation_results:
        ax2 = fig.add_subplot(gs[0, 1])
        event_importance = ablation_results['events']['importance_score']
        ax2.hist(event_importance, bins=50, color='crimson', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Importance Score (ΔP)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Ablation Event Importance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. IG Attribution Summary
    if 'summary' in ig_results:
        ax3 = fig.add_subplot(gs[0, 2])
        ig_summary = ig_results['summary']
        ax3.scatter(ig_summary['baseline_prob'], ig_summary['input_prob'], 
                   s=100, alpha=0.6, color='purple', edgecolor='black')
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Baseline P(Cancer)', fontsize=10)
        ax3.set_ylabel('Input P(Cancer)', fontsize=10)
        ax3.set_title('IG: Baseline vs Input Predictions', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Top Concepts by Ablation
    if 'concepts' in ablation_results:
        ax4 = fig.add_subplot(gs[1, :2])
        concept_agg = compare_concept_importance(ablation_results['concepts'], output_dir)
        top_concepts = concept_agg.head(20)
        
        colors = ['crimson' if x > 0 else 'steelblue' for x in top_concepts['mean_importance']]
        ax4.barh(range(len(top_concepts)), top_concepts['mean_importance'], 
                color=colors, edgecolor='black', alpha=0.7)
        ax4.set_yticks(range(len(top_concepts)))
        ax4.set_yticklabels([t[:25] for t in top_concepts['text']], fontsize=8)
        ax4.set_xlabel('Mean Importance Score', fontsize=10)
        ax4.set_title('Top 20 Medical Concepts (Ablation)', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.invert_yaxis()
    
    # 5. Concept Category Comparison
    if 'concepts' in ablation_results:
        ax5 = fig.add_subplot(gs[1, 2])
        category_importance = ablation_results['concepts'].groupby('concept_category')['importance_score'].mean()
        category_importance = category_importance.sort_values(ascending=False)
        
        colors_cat = ['crimson' if x > 0 else 'steelblue' for x in category_importance.values]
        ax5.bar(range(len(category_importance)), category_importance.values, 
               color=colors_cat, edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(len(category_importance)))
        ax5.set_xticklabels(category_importance.index, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Mean Importance', fontsize=10)
        ax5.set_title('Importance by Category', fontsize=12, fontweight='bold')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Method Agreement Summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = "METHOD COMPARISON SUMMARY\n" + "="*60 + "\n\n"
    
    # LR summary
    if 'weight_analysis' in lr_results:
        stats = lr_results['weight_analysis']['statistics']
        summary_text += "LR Coefficients:\n"
        summary_text += f"  • Mean weight: {stats['mean']:.6f}\n"
        summary_text += f"  • Weight range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
        summary_text += f"  • Sparsity (<0.01): {stats['sparsity_1e-2']*100:.1f}%\n\n"
    
    # Ablation summary
    if 'concepts' in ablation_results:
        concepts = ablation_results['concepts']
        n_concepts = len(concepts['concept_code'].unique())
        mean_importance = concepts['importance_score'].mean()
        summary_text += "Feature Ablation:\n"
        summary_text += f"  • Concepts analyzed: {n_concepts}\n"
        summary_text += f"  • Mean importance: {mean_importance:.6f}\n"
        summary_text += f"  • Positive (→cancer): {(concepts['importance_score'] > 0).sum()}\n"
        summary_text += f"  • Negative (→control): {(concepts['importance_score'] < 0).sum()}\n\n"
    
    # IG summary
    if 'summary' in ig_results:
        ig_summary = ig_results['summary']
        mean_completeness = ig_summary['completeness_score'].mean()
        summary_text += "Integrated Gradients:\n"
        summary_text += f"  • Patients analyzed: {len(ig_summary)}\n"
        summary_text += f"  • Mean completeness: {mean_completeness:.4f}\n"
        summary_text += f"  • Avg prediction change: {(ig_summary['input_prob'] - ig_summary['baseline_prob']).mean():.4f}\n\n"
    
    summary_text += "="*60 + "\n"
    summary_text += "KEY INSIGHTS:\n"
    summary_text += "• Compare top features across methods for consensus\n"
    summary_text += "• Look for features that rank highly in multiple methods\n"
    summary_text += "• Investigate disagreements for potential spurious correlations\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(os.path.join(output_dir, 'method_comparison_overview.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_report(
    lr_results: Dict,
    ablation_results: Dict,
    ig_results: Dict,
    output_dir: str
) -> None:
    """Create a detailed comparison report."""
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INTERPRETABILITY METHODS COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Method 1: LR Weights
        f.write("METHOD 1: LOGISTIC REGRESSION COEFFICIENTS\n")
        f.write("-"*80 + "\n")
        if 'weight_analysis' in lr_results:
            stats = lr_results['weight_analysis']['statistics']
            f.write(f"Global Analysis:\n")
            f.write(f"  Mean weight:        {stats['mean']:.6f}\n")
            f.write(f"  Std weight:         {stats['std']:.6f}\n")
            f.write(f"  Weight range:       [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            f.write(f"  Sparsity (<1e-3):   {stats['sparsity_1e-3']*100:.1f}%\n")
            f.write(f"  Sparsity (<1e-2):   {stats['sparsity_1e-2']*100:.1f}%\n\n")
            
            f.write("Top 10 Dimensions → Cancer:\n")
            for i, (dim, weight) in enumerate(lr_results['weight_analysis']['top_positive_dims'][:10], 1):
                f.write(f"  {i:2d}. Dim {dim:4d}: {weight:+.6f}\n")
            
            f.write("\nTop 10 Dimensions → Control:\n")
            for i, (dim, weight) in enumerate(lr_results['weight_analysis']['top_negative_dims'][:10], 1):
                f.write(f"  {i:2d}. Dim {dim:4d}: {weight:+.6f}\n")
        else:
            f.write("  No LR results available.\n")
        
        f.write("\n\n")
        
        # Method 2: Feature Ablation
        f.write("METHOD 2: FEATURE ABLATION\n")
        f.write("-"*80 + "\n")
        if 'concepts' in ablation_results:
            concepts = ablation_results['concepts']
            f.write(f"Concepts Analyzed: {len(concepts)}\n")
            f.write(f"Unique Concepts:   {concepts['concept_code'].nunique()}\n\n")
            
            concept_agg = compare_concept_importance(ablation_results['concepts'], output_dir)
            
            f.write("Top 15 Concepts → Cancer (Positive Importance):\n")
            top_pos = concept_agg[concept_agg['mean_importance'] > 0].head(15)
            for i, row in enumerate(top_pos.itertuples(), 1):
                f.write(f"  {i:2d}. {row.text[:40]:40s}  {row.mean_importance:+.6f} ({row.category})\n")
            
            f.write("\nTop 15 Concepts → Control (Negative Importance):\n")
            top_neg = concept_agg[concept_agg['mean_importance'] < 0].head(15)
            for i, row in enumerate(top_neg.itertuples(), 1):
                f.write(f"  {i:2d}. {row.text[:40]:40s}  {row.mean_importance:+.6f} ({row.category})\n")
        else:
            f.write("  No ablation results available.\n")
        
        f.write("\n\n")
        
        # Method 3: Integrated Gradients
        f.write("METHOD 3: INTEGRATED GRADIENTS\n")
        f.write("-"*80 + "\n")
        if 'summary' in ig_results:
            ig_summary = ig_results['summary']
            f.write(f"Patients Analyzed:       {len(ig_summary)}\n")
            f.write(f"Mean Baseline P(Cancer): {ig_summary['baseline_prob'].mean():.4f}\n")
            f.write(f"Mean Input P(Cancer):    {ig_summary['input_prob'].mean():.4f}\n")
            f.write(f"Mean Completeness:       {ig_summary['completeness_score'].mean():.4f}\n")
            f.write(f"Mean Attribution:        {ig_summary['attribution_mean'].mean():.6f}\n")
            f.write(f"Mean Attribution Std:    {ig_summary['attribution_std'].mean():.6f}\n")
        else:
            f.write("  No IG results available.\n")
        
        f.write("\n\n")
        
        # Recommendations
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        f.write("""
1. Look for CONSENSUS features that rank highly across multiple methods
2. Investigate DISAGREEMENTS between methods - may indicate spurious correlations
3. Validate top features with clinical experts
4. Check if known risk factors appear in top rankings
5. Compare against baseline/control group patterns

For Publication:
- Report all three methods for robustness
- Highlight consensus features with high confidence
- Discuss any method-specific findings
- Validate with held-out test set
""")
    
    print(f"\n  ✓ Comparison report saved to {report_path}")


def main(lr_dir: str, ablation_dir: str, ig_dir: str, output_dir: str):
    """Main comparison pipeline."""
    print("\n" + "=" * 80)
    print("INTERPRETABILITY METHODS COMPARISON")
    print("=" * 80)
    print(f"\nLR Results:       {lr_dir}")
    print(f"Ablation Results: {ablation_dir}")
    print(f"IG Results:       {ig_dir}")
    print(f"Output:           {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading results...")
    lr_results = load_lr_results(lr_dir)
    ablation_results = load_ablation_results(ablation_dir)
    ig_results = load_ig_results(ig_dir)
    
    print(f"  LR: {len(lr_results)} result types loaded")
    print(f"  Ablation: {len(ablation_results)} result types loaded")
    print(f"  IG: {len(ig_results)} result types loaded")
    
    # Create comparison visualizations
    print("\nCreating comparison visualizations...")
    visualize_method_comparison(lr_results, ablation_results, ig_results, output_dir)
    
    # Create detailed report
    print("Generating comparison report...")
    create_comparison_report(lr_results, ablation_results, ig_results, output_dir)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - method_comparison_overview.png : Visual comparison of all methods")
    print(f"  - comparison_report.txt          : Detailed text report")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare interpretability methods: LR, Ablation, and IG"
    )
    parser.add_argument(
        "--lr_dir",
        type=str,
        required=True,
        help="Directory with LR coefficient analysis results"
    )
    parser.add_argument(
        "--ablation_dir",
        type=str,
        required=True,
        help="Directory with feature ablation results"
    )
    parser.add_argument(
        "--ig_dir",
        type=str,
        required=True,
        help="Directory with Integrated Gradients results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./method_comparison",
        help="Directory to save comparison results"
    )
    
    args = parser.parse_args()
    
    main(
        lr_dir=args.lr_dir,
        ablation_dir=args.ablation_dir,
        ig_dir=args.ig_dir,
        output_dir=args.output_dir
    )

