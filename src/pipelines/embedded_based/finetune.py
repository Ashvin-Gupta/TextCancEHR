# src/experiments/finetune_embedded.py

"""
Fine-tuning script for embedding-based models on classification tasks.

This script performs supervised fine-tuning on pre-embedded EHR data:
- Can load pretrained checkpoints from pretraining phase
- Trains a classification head to predict cancer diagnosis
- Supports temporal cutoff (removing recent events before diagnosis)

Usage:
    python -m src.experiments.finetune_embedded --config configs/finetune_encoder_embedded.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from src.data.embedded_dataset import PreEmbeddedDataset
from src.data.embedding_collator import classification_collate_fn
from src.pipelines.embedded_based.models.transformer_decoder_embedded import TransformerDecoderEmbedded
from src.pipelines.embedded_based.models.transformer_encoder_embedded import TransformerEncoderEmbedded


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        # Move to device
        embeddings = batch['embeddings'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model({
            'embeddings': embeddings,
            'padding_mask': padding_mask
        })  # (B, num_classes)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue
            
            embeddings = batch['embeddings'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model({
                'embeddings': embeddings,
                'padding_mask': padding_mask
            })
            
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Calculate AUC if binary classification
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    if all_probs.shape[1] == 2:  # Binary classification
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 0.0
    else:
        auc = 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics


def main(config_path: str):
    """Main fine-tuning function."""
    print("=" * 80)
    print("Embedding-Based Model Fine-Tuning for Classification")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    train_dataset = PreEmbeddedDataset(
        data_dir=data_config['embedding_output_dir'],
        split='train',
        task='classification'
    )
    val_dataset = PreEmbeddedDataset(
        data_dir=data_config['embedding_output_dir'],
        split='tuning',
        task='classification'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=classification_collate_fn,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        shuffle=False,
        collate_fn=classification_collate_fn,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)
    
    if model_config['type'] == 'transformer_encoder_embedded':
        model = TransformerEncoderEmbedded(model_config)
    elif model_config['type'] == 'transformer_decoder_embedded':
        # For decoder, we need to add classification head
        model_config['add_classification_head'] = True
        model = TransformerDecoderEmbedded(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Load pretrained checkpoint if specified
    pretrained_checkpoint = training_config.get('pretrained_checkpoint', None)
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        print(f"\nLoading pretrained checkpoint from: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        
        # Try to load state dict (may have missing keys for classification head)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✓ Loaded pretrained weights (with potential missing keys for classification head)")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_config['type']}")
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )
    
    # Loss function with class weights if specified
    class_weights = training_config.get('class_weights', None)
    if class_weights:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=training_config.get('scheduler_patience', 3),
        verbose=True
    )
    
    # Create output directory
    output_dir = training_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to: {config_save_path}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting fine-tuning...")
    print("=" * 80)
    
    best_val_metric = 0.0  # Track best F1 score
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    
    for epoch in range(training_config['epochs']):
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'config': config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint (based on F1 score)
        if val_metrics['f1'] > best_val_metric:
            best_val_metric = val_metrics['f1']
            best_epoch = epoch + 1
            best_path = os.path.join(output_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with F1: {val_metrics['f1']:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Final summary
    print("\n" + "=" * 80)
    print("Fine-Tuning Complete!")
    print("=" * 80)
    print(f"Best F1 score: {best_val_metric:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune embedding-based models for classification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config)

