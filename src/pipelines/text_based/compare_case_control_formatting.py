"""
Compare formatting and punctuation differences between cases and controls.

This script loads case and control data and analyzes differences in:
- Punctuation marks
- Special characters
- Formatting patterns

Usage:
    python -m src.pipelines.text_based.compare_case_control_formatting --config_filepath path/to/config.yaml
"""

import argparse
import yaml
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
from src.data.unified_dataset import UnifiedEHRDataset
from tqdm import tqdm
import string


def count_characters(text: str, char_set: str) -> Dict[str, int]:
    """
    Count occurrences of specific characters in text.
    
    Args:
        text: Input text
        char_set: Set of characters to count
        
    Returns:
        Dictionary mapping character to count
    """
    counts = Counter()
    for char in text:
        if char in char_set:
            counts[char] += 1
    return dict(counts)


def analyze_text_formatting(texts: List[str], label: str) -> Dict:
    """
    Analyze formatting patterns in a list of texts.
    
    Args:
        texts: List of text strings
        label: Label for this group (e.g., "cases" or "controls")
        
    Returns:
        Dictionary of formatting statistics
    """
    # Character sets to analyze
    punctuation_marks = set(string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    whitespace_chars = {' ', '\t', '\n', '\r'}
    digits = set(string.digits)
    
    # Additional specific characters of interest
    special_chars = {'(', ')', '[', ']', '{', '}', '<', '>', ';', ':', ',', '.', '!', '?', '-', '_', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`', '"', "'"}
    
    stats = {
        'num_texts': len(texts),
        'total_chars': 0,
        'total_words': 0,
        'total_lines': 0,
        'punctuation': defaultdict(int),
        'special_chars': defaultdict(int),
        'whitespace': defaultdict(int),
        'digits': defaultdict(int),
        'avg_text_length': 0,
        'avg_word_length': 0,
        'char_frequencies': Counter(),
    }
    
    print(f"\nAnalyzing {label}...")
    for text in tqdm(texts, desc=f"Processing {label}"):
        # Basic stats
        stats['total_chars'] += len(text)
        stats['total_words'] += len(text.split())
        stats['total_lines'] += text.count('\n') + 1
        
        # Count each character type
        for char in text:
            stats['char_frequencies'][char] += 1
            
            if char in punctuation_marks:
                stats['punctuation'][char] += 1
            if char in special_chars:
                stats['special_chars'][char] += 1
            if char in whitespace_chars:
                stats['whitespace'][char] += 1
            if char in digits:
                stats['digits'][char] += 1
    
    # Calculate averages
    stats['avg_text_length'] = stats['total_chars'] / stats['num_texts'] if stats['num_texts'] > 0 else 0
    stats['avg_word_length'] = stats['total_chars'] / stats['total_words'] if stats['total_words'] > 0 else 0
    
    return stats


def compare_stats(cases_stats: Dict, controls_stats: Dict) -> pd.DataFrame:
    """
    Compare formatting statistics between cases and controls.
    
    Args:
        cases_stats: Statistics for cases
        controls_stats: Statistics for controls
        
    Returns:
        DataFrame with comparison results
    """
    comparisons = []
    
    # Compare basic stats
    basic_metrics = [
        ('Total Texts', 'num_texts'),
        ('Avg Text Length (chars)', 'avg_text_length'),
        ('Total Characters', 'total_chars'),
        ('Total Words', 'total_words'),
        ('Avg Word Length', 'avg_word_length'),
        ('Total Lines', 'total_lines'),
    ]
    
    for metric_name, key in basic_metrics:
        cases_val = cases_stats.get(key, 0)
        controls_val = controls_stats.get(key, 0)
        
        if isinstance(cases_val, (int, float)) and isinstance(controls_val, (int, float)):
            # Per-text normalization
            cases_per_text = cases_val / cases_stats['num_texts'] if cases_stats['num_texts'] > 0 else 0
            controls_per_text = controls_val / controls_stats['num_texts'] if controls_stats['num_texts'] > 0 else 0
            
            comparisons.append({
                'Metric': metric_name,
                'Cases (Total)': f"{cases_val:,.2f}" if isinstance(cases_val, float) else f"{cases_val:,}",
                'Controls (Total)': f"{controls_val:,.2f}" if isinstance(controls_val, float) else f"{controls_val:,}",
                'Cases (Per Text)': f"{cases_per_text:.2f}",
                'Controls (Per Text)': f"{controls_per_text:.2f}",
                'Difference (%)': f"{((cases_per_text - controls_per_text) / controls_per_text * 100 if controls_per_text > 0 else 0):.1f}%"
            })
    
    # Compare punctuation
    all_punct = set(cases_stats['punctuation'].keys()) | set(controls_stats['punctuation'].keys())
    for char in sorted(all_punct):
        cases_count = cases_stats['punctuation'].get(char, 0)
        controls_count = controls_stats['punctuation'].get(char, 0)
        
        cases_per_text = cases_count / cases_stats['num_texts']
        controls_per_text = controls_count / controls_stats['num_texts']
        
        if cases_count > 0 or controls_count > 0:
            comparisons.append({
                'Metric': f"'{char}' punctuation",
                'Cases (Total)': f"{cases_count:,}",
                'Controls (Total)': f"{controls_count:,}",
                'Cases (Per Text)': f"{cases_per_text:.2f}",
                'Controls (Per Text)': f"{controls_per_text:.2f}",
                'Difference (%)': f"{((cases_per_text - controls_per_text) / controls_per_text * 100 if controls_per_text > 0 else 0):.1f}%"
            })
    
    # Compare whitespace
    whitespace_map = {' ': 'space', '\t': 'tab', '\n': 'newline', '\r': 'carriage return'}
    for char, name in whitespace_map.items():
        cases_count = cases_stats['whitespace'].get(char, 0)
        controls_count = controls_stats['whitespace'].get(char, 0)
        
        cases_per_text = cases_count / cases_stats['num_texts']
        controls_per_text = controls_count / controls_stats['num_texts']
        
        if cases_count > 0 or controls_count > 0:
            comparisons.append({
                'Metric': f"'{name}' whitespace",
                'Cases (Total)': f"{cases_count:,}",
                'Controls (Total)': f"{controls_count:,}",
                'Cases (Per Text)': f"{cases_per_text:.2f}",
                'Controls (Per Text)': f"{controls_per_text:.2f}",
                'Difference (%)': f"{((cases_per_text - controls_per_text) / controls_per_text * 100 if controls_per_text > 0 else 0):.1f}%"
            })
    
    return pd.DataFrame(comparisons)


def main(config_path: str, max_samples: int = None):
    """
    Main function to compare formatting between cases and controls.
    
    Args:
        config_path: Path to YAML configuration file
        max_samples: Maximum number of samples per group (for testing)
    """
    print("=" * 80)
    print("Case vs Control Formatting Comparison")
    print("=" * 80)
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "time_lookup_file": data_config["time_lookup_filepath"],
        "format": 'text',
        "cutoff_months": data_config.get("cutoff_months", None),
        "max_sequence_length": None,
        "data_type": data_config.get('data_type', 'raw')
    }
    
    # Load train dataset (or change to test/val as needed)
    dataset = UnifiedEHRDataset(split="train", **dataset_args)
    print(f"Loaded {len(dataset)} total samples")
    
    # Separate into cases and controls
    case_texts = []
    control_texts = []
    
    print("\nSeparating cases and controls...")
    for i in tqdm(range(len(dataset)), desc="Loading samples"):
        if max_samples and len(case_texts) >= max_samples and len(control_texts) >= max_samples:
            break
            
        sample = dataset[i]
        if sample is None:
            continue
        
        text = sample['text']
        label = sample['label'].item() if hasattr(sample['label'], 'item') else sample['label']
        
        if label > 0:  # Case
            if not max_samples or len(case_texts) < max_samples:
                case_texts.append(text)
        else:  # Control
            if not max_samples or len(control_texts) < max_samples:
                control_texts.append(text)
    
    print(f"\nFound {len(case_texts)} cases and {len(control_texts)} controls")
    
    # Analyze both groups
    cases_stats = analyze_text_formatting(case_texts, "Cases")
    controls_stats = analyze_text_formatting(control_texts, "Controls")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    comparison_df = compare_stats(cases_stats, controls_stats)
    
    # Display results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save to CSV
    output_file = "case_control_formatting_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Show top differences
    print("\n" + "=" * 80)
    print("TOP 10 LARGEST DIFFERENCES (by percentage)")
    print("=" * 80)
    
    # Convert percentage strings to float for sorting
    comparison_df['Diff_Numeric'] = comparison_df['Difference (%)'].str.rstrip('%').astype(float)
    top_diffs = comparison_df.nlargest(10, 'Diff_Numeric', keep='all')
    top_diffs = top_diffs.drop('Diff_Numeric', axis=1)
    
    print("\n" + top_diffs.to_string(index=False))
    
    # Show some example texts
    print("\n" + "=" * 80)
    print("EXAMPLE TEXTS")
    print("=" * 80)
    
    print("\n--- CASE EXAMPLE (first 500 chars) ---")
    if case_texts:
        print(case_texts[0][:500])
    
    print("\n--- CONTROL EXAMPLE (first 500 chars) ---")
    if control_texts:
        print(control_texts[0][:500])
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare formatting between cases and controls")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per group (for testing, default: all)"
    )
    args = parser.parse_args()
    
    main(args.config_filepath, args.max_samples)
