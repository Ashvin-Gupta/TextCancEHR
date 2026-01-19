# src/training/classification_trainer.py

"""
Classification trainer for LLM-based binary classification.

Provides a wrapper model that adds a classification head on top of a frozen LLM,
and functions to train only the classification head.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from src.evaluations.visualisation import plot_classification_performance
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from transformers import EarlyStoppingCallback


class LLMClassifier(nn.Module):
    """
    Wrapper that adds a classification head on top of a (frozen) LLM.
    
    The LLM extracts hidden states, and we use the last non-padding token's
    hidden state as input to a simple linear classification head.
    
    Args:
        base_model: The pretrained LLM (with or without LoRA adapters)
        hidden_size: Hidden dimension of the LLM
        num_labels: Number of output classes (2 for binary classification)
        freeze_base: Whether to freeze the base LLM parameters
        trainable_param_keywords: Optional substrings for parameters that should remain trainable
        multi_label: If True, uses BCE loss (future multi-label support)
    """
    
    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_labels: int = 2,
        freeze_base: bool = True,
        trainable_param_keywords=None,
        multi_label: bool = False,
        tokenizer=None
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.trainable_param_keywords = trainable_param_keywords or []
        self.tokenizer = tokenizer

        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            print("  - Enabling gradient checkpointing for base model")
            self.base_model.gradient_checkpointing_enable()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("  - Froze all base model parameters")
        
        if self.trainable_param_keywords:
            reenabled = 0
            for name, param in self.base_model.named_parameters():
                if any(keyword in name for keyword in self.trainable_param_keywords):
                    param.requires_grad = True
                    reenabled += 1
           
        
        # Classification head: hidden_size -> num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        print(f"  - Added classification head: {hidden_size} -> {num_labels}")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Delegate gradient checkpointing enable to the base model.
        """
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """
        Helper often needed by Trainer to verify model compatibility.
        """
        return self.base_model.get_input_embeddings()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            labels: (batch_size,) ground truth labels
        
        Returns:
            Dict with:
                - loss: scalar loss (if labels provided)
                - logits: (batch_size, num_labels) classification logits
                - hidden_states: (batch_size, hidden_size) extracted features ''SHOULD NOT DO THIS NOW''
        """
        backbone = getattr(self.base_model, "model", self.base_model)

        # Get hidden states from the LLM
        # Note: Most LLMs return a tuple or a special object
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract the last layer's hidden states
        # Shape: (batch_size, seq_len, hidden_size)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Get the last non-padding token's hidden state for each sequence
        # The attention_mask is 1 for real tokens, 0 for padding
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_size = hidden_states.size(0)

        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]

        
        try:
            is_main_process = torch.distributed.get_rank() == 0
        except:
            is_main_process = True  # Not using distributed training

        # Print for first batch or randomly 1% of the time
        if is_main_process and (not hasattr(self, '_debug_count') or self._debug_count < 10 or torch.rand(1).item() < 0.01):
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
                self._eos_stats = {'case': {'correct': 0, 'pad': 0, 'other': 0}, 
                                'control': {'correct': 0, 'pad': 0, 'other': 0}}
            self._debug_count += 1
            
            for i in range(batch_size):
                seq_len = sequence_lengths[i].item()
                token_at_position = input_ids[i, seq_len].item()
                
                # Determine if case or control
                label_val = labels[i].item() if labels is not None else -1
                is_case = label_val > 0
                case_type = 'case' if is_case else 'control'
                
            #     if self.tokenizer is not None:
            #         token_text = self.tokenizer.decode([token_at_position])
            #         eos_id = self.tokenizer.eos_token_id
            #         pad_id = self.tokenizer.pad_token_id
            #         is_eos = (token_at_position == eos_id)
            #         is_pad = (token_at_position == pad_id)
                    
            #         # Update statistics
            #         if is_eos:
            #             self._eos_stats[case_type]['correct'] += 1
            #         elif is_pad:
            #             self._eos_stats[case_type]['pad'] += 1
            #         else:
            #             self._eos_stats[case_type]['other'] += 1
                    
            #         # Print detailed info for first 10 batches
            #         if self._debug_count <= 10:
            #             print(f"\n{'='*60}")
            #             print(f"  Patient {i} Debug (Batch #{self._debug_count}) - {case_type.upper()}")
            #             print(f"{'='*60}")
            #             print(f"    - Label: {label_val} ({'Case' if is_case else 'Control'})")
            #             print(f"    - Sequence length (from attention_mask): {seq_len}")
            #             print(f"    - Token at position {seq_len}: ID={token_at_position}, text='{token_text}'")
            #             print(f"    - Is EOS? {is_eos} (EOS ID={eos_id})")
            #             print(f"    - Is PAD? {is_pad} (PAD ID={pad_id})")
            #             print(f"    - Attention mask sum: {attention_mask[i].sum().item()}")
            #             print(f"    - Input IDs shape: {input_ids[i].shape}")
            #             print(f"    - Attention mask at position {seq_len}: {attention_mask[i, seq_len].item()}")
                        
            #             # Find where EOS actually is
            #             eos_positions = (input_ids[i] == eos_id).nonzero(as_tuple=True)[0]
            #             if len(eos_positions) > 0:
            #                 print(f"    - EOS token found at positions: {eos_positions.tolist()}")
            #                 for eos_pos in eos_positions:
            #                     eos_pos_val = eos_pos.item()
            #                     eos_mask = attention_mask[i, eos_pos_val].item()
            #                     print(f"        pos {eos_pos_val}: mask={eos_mask}")
            #             else:
            #                 print(f"    - ⚠️  WARNING: No EOS token found in sequence!")
                        
            #             # Show last 5 tokens
            #             last_5_positions = max(0, seq_len - 4)
            #             print(f"    - Last 5 tokens:")
            #             for pos in range(last_5_positions, min(seq_len + 2, input_ids.shape[1])):
            #                 tid = input_ids[i, pos].item()
            #                 ttext = self.tokenizer.decode([tid])
            #                 mask_val = attention_mask[i, pos].item()
            #                 is_this_eos = (tid == eos_id)
            #                 marker = " <-- EOS!" if is_this_eos else ""
            #                 print(f"        pos {pos}: ID={tid}, text='{ttext}', mask={mask_val}{marker}")
            #             print(f"{'='*60}\n")
            
            # # Print statistics every 50 batches or at batch 10
            # if self._debug_count % 50 == 0 or self._debug_count == 10:
            #     print(f"\n{'#'*70}")
            #     print(f"  EOS TOKEN EXTRACTION STATISTICS (After {self._debug_count} batches)")
            #     print(f"{'#'*70}")
                
            #     for case_type in ['case', 'control']:
            #         stats = self._eos_stats[case_type]
            #         total = sum(stats.values())
            #         if total > 0:
            #             print(f"\n  {case_type.upper()}S:")
            #             print(f"    - Correct (EOS):  {stats['correct']:4d} ({stats['correct']/total*100:5.1f}%)")
            #             print(f"    - Padding (PAD):  {stats['pad']:4d} ({stats['pad']/total*100:5.1f}%)")
            #             print(f"    - Other:          {stats['other']:4d} ({stats['other']/total*100:5.1f}%)")
            #             print(f"    - Total:          {total:4d}")
            #     print(f"{'#'*70}\n")
        
        # Pass through classification head
        # Shape: (batch_size, num_labels)
        logits = self.classifier(last_hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.multi_label:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            # 'hidden_states': last_hidden_states
        }
    


def compute_metrics(eval_pred):
    """
    Compute classification metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
    
    Returns:
        Dict of metric names to values
    """
    predictions, labels = eval_pred
    
    # Handle case where predictions is a tuple/dict (model returns multiple outputs)
    # Extract just the logits (first element if tuple, or 'logits' key if dict)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Usually logits are first
    elif isinstance(predictions, dict):
        predictions = predictions['logits']
    
    # Now predictions should be shape (batch_size, num_labels)
    # Get predicted class (argmax of logits)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # Calculate AUROC using softmax probabilities
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()[:, 1]
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        # If only one class present in labels
        auroc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def run_classification_training(
    config: Dict[str, Any],
    model: nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    collate_fn
):
    """
    Run the classification training loop.
    
    Args:
        config: Configuration dict with model, training, and data settings
        model: LLMClassifier model
        tokenizer: Tokenizer (for saving with model)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        collate_fn: Data collator function
    """
    training_config = config['training']
    model_config = config['model']
    wandb_config = config.get('wandb', {})
    multi_label_task = bool(training_config.get('multi_label', False))
    
    if multi_label_task:
        print("Multi-label training mode is flagged, but metrics/plots currently assume binary classification.")

    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        run_name=wandb_config.get('run_name', 'llm-classifier'),
        report_to="wandb" if wandb_config.get('enabled', False) else "none",
        
        # Training hyperparameters
        num_train_epochs=int(training_config.get('epochs', 10)),
        per_device_train_batch_size=int(training_config.get('batch_size', 8)),
        per_device_eval_batch_size=int(training_config.get('eval_batch_size', 8)),
        learning_rate=float(training_config.get('learning_rate', 1e-3)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=int(training_config.get('warmup_steps', 100)),
        
        # Gradient settings
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 1)),
        gradient_checkpointing=True,

        # Multi-GPU settings (NEW)
        ddp_find_unused_parameters=False,  # Set to False for efficiency
        dataloader_num_workers=training_config.get('dataloader_num_workers', 8),
            
        # Precision
        fp16=bool(training_config.get('fp16', False)),
        bf16=bool(training_config.get('bf16', True)),
        
        # Logging and evaluation
        logging_steps=int(training_config.get('logging_steps', 10)),
        evaluation_strategy="steps",
        eval_steps=int(training_config.get('eval_steps', 100)),
        save_strategy="steps",
        save_steps=int(training_config.get('save_steps', 500)),
        save_total_limit=int(training_config.get('save_total_limit', 2)),
        
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Other
        remove_unused_columns=False,  # Keep our custom columns
    )
    
    # Create trainer
    print("\nInitializing Trainer...")
    callbacks = []
    early_stopping_patience = training_config.get('early_stopping_patience', None)
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=training_config.get('early_stopping_threshold', 0.0)
        ))
        print(f"  - Early stopping enabled with patience={early_stopping_patience}")
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting classification training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    print(f"  - Model saved to: {final_model_path}")
    
    # Run final evaluation on validation set
    print("\nRunning final evaluation on validation set...")
    eval_results = trainer.evaluate()
    print("\nFinal Validation Results:")
    print("="*80)
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    print("="*80)

    print("\nGenerating detailed ROC and Precision-Recall curves for validation set...")
    
    # 1. Get raw predictions (logits) for every patient in validation set, evaluate() only gives averages; predict() gives the actual scores
    pred_output = trainer.predict(val_dataset)
    
    # 2. Convert logits to probabilities
    # logits = pred_output.predictions[0]
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    
    if multi_label_task:
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
    else:
        # We take column 1 because that is the probability of "Cancer"
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    labels = pred_output.label_ids
    
    # 3. Create the folder for plots
    plot_output_dir = os.path.join(training_config['output_dir'], "plots_validation")
    
    # 4. Call your helper function to draw the graphs
    if multi_label_task:
        print("Skipping ROC/PR plots for multi-label configuration (not yet supported).")
    else:
        plot_classification_performance(labels, probs, plot_output_dir)
    
    # Run evaluation on test set
    print("\nRunning evaluation on test set...")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("\nFinal Test Results:")
    print("="*80)
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print("="*80)
    
    print("\nGenerating detailed ROC and Precision-Recall curves for test set...")
    
    # Get predictions for test set
    test_pred_output = trainer.predict(test_dataset)
    
    # Convert logits to probabilities
    # test_logits = test_pred_output.predictions[0]
    test_logits = test_pred_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    
    if multi_label_task:
        test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    else:
        test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
    test_labels = test_pred_output.label_ids
    
    # Create the folder for test plots
    test_plot_output_dir = os.path.join(training_config['output_dir'], "plots_test")
    
    # Generate test set plots
    if multi_label_task:
        print("Skipping ROC/PR plots for multi-label configuration (not yet supported).")
    else:
        plot_classification_performance(test_labels, test_probs, test_plot_output_dir)
    
    return trainer, eval_results


