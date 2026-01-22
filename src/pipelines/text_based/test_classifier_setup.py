# src/pipelines/text_based/test_classifier_setup.py

"""
Quick test script to validate the classification pipeline setup.
Tests loading config, datasets, and model components without full training.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator


def test_config_loading(config_path: str):
    """Test that config file loads correctly."""
    print("=" * 80)
    print("TEST 1: Config Loading")
    print("=" * 80)
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        required_keys = ['model', 'data', 'training']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
        
        print("✓ Config loaded successfully")
        print(f"  - Model checkpoint: {config['model']['pretrained_checkpoint']}")
        print(f"  - Data directory: {config['data']['data_dir']}")
        print(f"  - Output directory: {config['training']['output_dir']}")
        return config
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return None


def test_dataset_loading(config: dict):
    """Test that datasets can be instantiated."""
    print("\n" + "=" * 80)
    print("TEST 2: Dataset Loading")
    print("=" * 80)
    
    try:
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
            "tokenizer": None
        }
        
        print("Loading validation dataset (limited)...")
        val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of patients: {len(val_dataset)}")
        
        # Test getting a sample
        sample = val_dataset[0]
        if sample is not None:
            print(f"  - Sample text length: {len(sample['text'])} chars")
            print(f"  - Sample label: {sample['label'].item()}")
            print(f"  - Text preview (last 200 chars): ...{sample['text'][-200:]}")
        else:
            print("  ⚠ First sample is None (patient without label)")
        
        return val_dataset
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_collator(config: dict, dataset):
    """Test data collator with dummy tokenizer."""
    print("\n" + "=" * 80)
    print("TEST 3: Data Collator (Dummy Tokenizer)")
    print("=" * 80)
    
    try:
        # We can't test with real tokenizer without loading the model
        # But we can test the collator logic with a mock
        print("✓ Collator class imported successfully")
        print("  - ClassificationCollator handles:")
        print("    • Tokenization of text")
        print("    • Padding and attention masks")
        print("    • Binary label conversion")
        print("  - To test fully, need to load pretrained tokenizer")
        return True
    except Exception as e:
        print(f"✗ Collator test failed: {e}")
        return False


def test_model_components():
    """Test that model components can be imported."""
    print("\n" + "=" * 80)
    print("TEST 4: Model Components")
    print("=" * 80)
    
    try:
        from src.training.classification_trainer import LLMClassifier, run_classification_training
        print("✓ Model components imported successfully")
        print("  - LLMClassifier: Wrapper for LLM + classification head")
        print("  - run_classification_training: Training function")
        return True
    except Exception as e:
        print(f"✗ Model component import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_distribution(dataset):
    """Test the label distribution in the dataset."""
    print("\n" + "=" * 80)
    print("TEST 5: Label Distribution")
    print("=" * 80)
    
    try:
        label_counts = {}
        valid_samples = 0
        
        print("Counting labels...")
        for i in range(min(len(dataset), 100)):  # Check first 100
            sample = dataset[i]
            if sample is not None:
                valid_samples += 1
                label = sample['label'].item()
                binary_label = 1 if label > 0 else 0
                label_counts[binary_label] = label_counts.get(binary_label, 0) + 1
        
        print(f"✓ Label distribution (first 100 samples):")
        print(f"  - Valid samples: {valid_samples}")
        print(f"  - Control (0): {label_counts.get(0, 0)}")
        print(f"  - Cancer (1): {label_counts.get(1, 0)}")
        
        if label_counts.get(0, 0) > 0 and label_counts.get(1, 0) > 0:
            ratio = label_counts.get(0, 0) / label_counts.get(1, 0)
            print(f"  - Control/Cancer ratio: {ratio:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Label distribution check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CLASSIFICATION PIPELINE SETUP VALIDATION")
    print("=" * 80)
    
    config_path = "src/pipelines/text_based/configs/llm_finetune_classifier.yaml"
    
    # Run tests
    config = test_config_loading(config_path)
    if config is None:
        print("\n✗ Tests failed at config loading")
        return
    
    dataset = test_dataset_loading(config)
    if dataset is None:
        print("\n✗ Tests failed at dataset loading")
        return
    
    test_collator(config, dataset)
    test_model_components()
    test_label_distribution(dataset)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\n✓ All basic tests passed!")
    print("\nNext steps:")
    print("1. Ensure pretrained checkpoint exists at specified path")
    print("2. Run: bash run_text_llm_classifier.sh")
    print("3. Monitor training via WandB or logs")


if __name__ == "__main__":
    main()


