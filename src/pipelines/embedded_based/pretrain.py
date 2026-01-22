# src/experiments/pretrain_embedded.py

"""
Pretraining script for embedding-based models.

This script performs unsupervised pretraining on pre-embedded EHR data:
- Decoder models: Autoregressive next-event prediction
- Encoder models: Can be extended for MLM (currently just goes to fine-tuning)

Usage:
    python -m src.experiments.pretrain_embedded --config configs/pretrain_decoder_embedded.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.data.embedded_dataset import PreEmbeddedDataset
from src.data.embedding_collator import get_collate_fn
from src.pipelines.embedded_based.models.transformer_decoder_embedded import TransformerDecoderEmbedded
from src.pipelines.embedded_based.models.transformer_encoder_embedded import TransformerEncoderEmbedded


def extract_text(base_dataset, tokenizer):
        """Extracts all valid text narratives and adds EOS token."""
        text_list = []
        # Use eos_token if it exists, otherwise use an empty string
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        
        print(f"  - Processing {len(base_dataset)} patients...")
        # We iterate through the base_dataset to get the text
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item is not None:
                # item['text'] is the narrative from UnifiedEHRDataset
                text_list.append(item['text'] + ', ' + eos_token)
        print(f"  - Extracted {len(text_list)} valid narratives.")
        return text_list
        
def train_epoch(model, dataloader, optimizer, criterion, device, task):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        # Move to device
        if task == 'autoregressive':
            input_embeddings = batch['input_embeddings'].to(device)
            target_token_ids = batch['target_token_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model({
                'input_embeddings': input_embeddings,
                'padding_mask': padding_mask
            })  # (B, T, vocab_size)
            print(f'logits shape: {logits.shape}')
            print(f'target_token_ids shape: {target_token_ids.shape}')
            # Compute loss (ignore padding tokens)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_token_ids.reshape(-1)
            )
            
            # Count valid tokens
            num_tokens = padding_mask.sum().item()
        else:
            raise ValueError(f"Unsupported task for pretraining: {task}")
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss


def evaluate(model, dataloader, criterion, device, task):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue
            
            if task == 'autoregressive':
                input_embeddings = batch['input_embeddings'].to(device)
                target_token_ids = batch['target_token_ids'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                
                logits = model({
                    'input_embeddings': input_embeddings,
                    'padding_mask': padding_mask
                })
                
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_token_ids.view(-1)
                )
                
                num_tokens = padding_mask.sum().item()
            else:
                raise ValueError(f"Unsupported task for pretraining: {task}")
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def main(config_path: str):
    """Main pretraining function."""
    print("=" * 80)
    print("Embedding-Based Model Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']

    # Set up WandB
    wandb_config = config.get('wandb', {})

    default_run_name = (
        f"{model_config['type']}-pretrain-{data_config['cutoff_months']}month-cutoff"
        f"_lr{training_config['learning_rate']}"
        f"_bs{training_config['batch_size']}"
        f"_wd{training_config['weight_decay']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if wandb_config.get('enabled', False):
        wandb.init(
            project=wandb_config.get("project", "embedded-pretraining"),
            config=config,
            name=wandb_config.get("run_name", default_run_name)
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine task
    task = training_config.get('task', 'autoregressive')
    print(f"Task: {task}")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    train_dataset = PreEmbeddedDataset(
        data_dir=data_config['embedding_output_dir'],
        split='train',
        task=task,
        max_sequence_length=model_config['context_length']
    )
    val_dataset = PreEmbeddedDataset(
        data_dir=data_config['embedding_output_dir'],
        split='tuning',
        task=task,
        max_sequence_length=model_config['context_length']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    collate_fn = get_collate_fn(task)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)
    
    if model_config['type'] == 'transformer_decoder_embedded':
        model = TransformerDecoderEmbedded(model_config)
    elif model_config['type'] == 'transformer_encoder_embedded':
        model = TransformerEncoderEmbedded(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_config['type']}")
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        betas=(0.9, 0.999)
    )
    
    # Loss function (ignore padding token ID = 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config['epochs']),
        eta_min=training_config.get('min_learning_rate', 1e-6)
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
    print("Starting pretraining...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(int(training_config['epochs'])):
        print(f"\nEpoch {epoch + 1}/{training_config['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, task)
        
        # Evaluate
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, device, task)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if wandb_config.get('enabled', False):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": current_lr
            })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(output_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = os.path.join(output_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % training_config.get('save_every', 10) == 0:
            periodic_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save(checkpoint, periodic_path)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Pretraining Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {output_dir}")

    if wandb_config.get('enabled', False):
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain embedding-based models")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config_filepath)

