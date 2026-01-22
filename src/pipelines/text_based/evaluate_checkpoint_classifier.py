import argparse
import yaml
import os
import torch
import numpy as np
from transformers import Trainer, TrainingArguments

# Import your existing modules
from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier, compute_metrics
from src.evaluations.visualisation import plot_classification_performance
from src.training.utils import load_LoRA_model
from src.training.utils import seed_all
def main(config_path: str):
    
    # Set seed
    seed_all(42)
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    
    data_config = config['data']
    model_config = config['model']
    finetuned_checkpoint = model_config['finetune_checkpoint']
    print(f"Evaluating Checkpoint: {finetuned_checkpoint}")
    print("=" * 80)
        
    # 2. Initialize Base Model using your Helper
    # This handles the resizing, LoRA adapters, and tokenizer setup automatically
    print("\nLoading base model architecture...")
    base_model, tokenizer = load_LoRA_model(config)
    
    # 3. Initialize the Classifier Wrapper
    # We wrap the loaded base model just like in training
    print("Initializing classifier wrapper...")
    model = LLMClassifier(
        base_model=base_model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=True
    )
    
    # 4. Load the TRAINED Classifier Weights
    # load_LoRA_model gave us the structure; now we overwrite it with the 
    # specific weights learned during classification training (including the head)
    print(f"Loading trained classifier weights from {finetuned_checkpoint}...")
    
    bin_path = os.path.join(finetuned_checkpoint, "pytorch_model.bin")
    safe_path = os.path.join(finetuned_checkpoint, "model.safetensors")
    
    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find model weights in {finetuned_checkpoint}")
        
    model.load_state_dict(state_dict, strict=False) 
    print("✓ Classifier weights loaded successfully.")

    # 5. Load Validation Dataset (Same as before)
    print("\nLoading validation dataset...")
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
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    test_dataset = UnifiedEHRDataset(split="held_out", **dataset_args)
    
    # 6. Create Collator
    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config['max_length'],
        binary_classification=True
    )
    
    # 7. Setup Trainer (Inference Mode)
    training_args = TrainingArguments(
        output_dir=finetuned_checkpoint,
        per_device_eval_batch_size=8,
        report_to="none",
        bf16=True, # Ensure this matches your hardware
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # 8. Run Prediction on Validation Set
    print("\nRunning prediction on validation set...")
    pred_output = trainer.predict(val_dataset)
    
    # Print Metrics
    print("\nValidation Set Metrics:")
    for key, val in pred_output.metrics.items():
        print(f"{key}: {val:.4f}")
        
    # 9. Generate Plots for Validation Set
    print("\nGenerating PR and ROC curves for validation set...")
    predictions = pred_output.predictions
    logits = predictions[0]
    labels = pred_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    
    plot_dir = os.path.join(finetuned_checkpoint, "evaluation_plots_validation")
    plot_classification_performance(labels, probs, plot_dir)
    
    # 10. Run Prediction on Test Set
    print("\nRunning prediction on test set...")
    test_pred_output = trainer.predict(test_dataset)
    
    # Print Metrics
    print("\nTest Set Metrics:")
    for key, val in test_pred_output.metrics.items():
        print(f"{key}: {val:.4f}")
        
    # 11. Generate Plots for Test Set
    print("\nGenerating PR and ROC curves for test set...")
    test_predictions = test_pred_output.predictions
    test_logits = test_predictions[0]
    test_labels = test_pred_output.label_ids
    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    
    test_plot_dir = os.path.join(finetuned_checkpoint, "evaluation_plots_test")
    plot_classification_performance(test_labels, test_probs, test_plot_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    main(args.config_filepath)