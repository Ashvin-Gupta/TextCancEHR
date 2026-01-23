# src/training/motor_trainer.py

"""
MOTOR-style Piecewise Exponential Time-to-Event Model for LLM-based survival analysis.

Based on: "MOTOR: A Time-To-Event Foundation Model For Structured Medical Records"
(Steinberg et al., ICLR 2024)

This implementation uses the EOS token representation from a pretrained LLM
and applies a piecewise exponential head for time-to-event prediction.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback


class PiecewiseExponentialHead(nn.Module):
    """
    Piecewise Exponential Head following MOTOR paper.
    
    Approximates the time-to-event distribution with a piecewise exponential function.
    For each piece p, estimates a hazard λ_p.
    
    Uses low-rank factorization: log(λ_p) = M_p · β
    where:
    - M_p is the time-dependent transformation of patient representation for piece p
    - β is the task embedding (for single task, just learnable parameters)
    
    Args:
        hidden_size: Dimension of input representation (LLM hidden size)
        num_pieces: Number of time pieces P
        piece_boundaries: List of (start, end) tuples defining piece boundaries in days
        intermediate_dim: Intermediate dimension for low-rank factorization
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_pieces: int = 6,
        piece_boundaries: Optional[List[Tuple[float, float]]] = None,
        intermediate_dim: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pieces = num_pieces
        self.intermediate_dim = intermediate_dim
        
        # Default piece boundaries (in days): 0-30, 30-90, 90-180, 180-365, 365-730, 730+
        if piece_boundaries is None:
            self.piece_boundaries = [
                (0, 30),      # 0-1 month
                (30, 90),     # 1-3 months
                (90, 180),    # 3-6 months
                (180, 365),   # 6-12 months
                (365, 730),   # 1-2 years
                (730, float('inf'))  # 2+ years
            ]
        else:
            self.piece_boundaries = piece_boundaries
            
        assert len(self.piece_boundaries) == num_pieces, \
            f"Number of boundaries ({len(self.piece_boundaries)}) must match num_pieces ({num_pieces})"
        
        # Low-rank transformation: R_i -> M_ip (time-dependent state per piece)
        # Shape: hidden_size -> intermediate_dim * num_pieces
        self.state_projection = nn.Linear(hidden_size, intermediate_dim * num_pieces)
        
        # Task embedding β (for single task)
        # Shape: intermediate_dim
        self.task_embedding = nn.Parameter(torch.randn(intermediate_dim) * 0.01)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.state_projection.weight)
        nn.init.zeros_(self.state_projection.bias)
        
    def forward(self, patient_repr: torch.Tensor) -> torch.Tensor:
        """
        Compute log-hazards for each piece.
        
        Args:
            patient_repr: (batch_size, hidden_size) - EOS token representation
            
        Returns:
            log_hazards: (batch_size, num_pieces) - log(λ_p) for each piece
        """
        batch_size = patient_repr.size(0)
        
        # Project to time-dependent states: (batch_size, intermediate_dim * num_pieces)
        M = self.state_projection(patient_repr)
        
        # Reshape to (batch_size, num_pieces, intermediate_dim)
        M = M.view(batch_size, self.num_pieces, self.intermediate_dim)
        
        # Compute log-hazards: log(λ_p) = M_p · β
        # (batch_size, num_pieces, intermediate_dim) @ (intermediate_dim,) -> (batch_size, num_pieces)
        log_hazards = torch.einsum('bpi,i->bp', M, self.task_embedding)
        
        return log_hazards
    
    def get_piece_index(self, time: torch.Tensor) -> torch.Tensor:
        """
        Get the piece index for a given time.
        
        Args:
            time: (batch_size,) - time values in days
            
        Returns:
            piece_idx: (batch_size,) - piece index for each time
        """
        device = time.device
        piece_idx = torch.zeros_like(time, dtype=torch.long)
        
        for p, (start, end) in enumerate(self.piece_boundaries):
            mask = (time >= start) & (time < end)
            piece_idx[mask] = p
            
        return piece_idx
    
    def compute_time_in_pieces(
        self, 
        event_time: torch.Tensor, 
        event_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time spent in each piece (U_ijkp) and event indicators (δ_ijkp).
        
        Args:
            event_time: (batch_size,) - time to event or censoring in days
            event_indicator: (batch_size,) - 1 if event occurred, 0 if censored
            
        Returns:
            U: (batch_size, num_pieces) - time in each piece
            delta: (batch_size, num_pieces) - event indicator for each piece
        """
        batch_size = event_time.size(0)
        device = event_time.device
        
        U = torch.zeros(batch_size, self.num_pieces, device=device)
        delta = torch.zeros(batch_size, self.num_pieces, device=device)
        
        for p, (start, end) in enumerate(self.piece_boundaries):
            # Time spent in this piece
            # min(event_time, end) - start, but only if event_time >= start
            end_tensor = torch.tensor(end if end != float('inf') else 1e9, device=device)
            time_in_piece = torch.clamp(
                torch.minimum(event_time, end_tensor) - start,
                min=0
            )
            # Only count if patient reached this piece
            reached_piece = event_time >= start
            U[:, p] = time_in_piece * reached_piece.float()
            
            # Event indicator for this piece: event occurred AND event is in this piece
            event_in_piece = (event_time >= start) & (event_time < end_tensor) & (event_indicator == 1)
            delta[:, p] = event_in_piece.float()
            
        return U, delta


def piecewise_exponential_loss(
    log_hazards: torch.Tensor,
    U: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute piecewise exponential negative log-likelihood loss.
    
    From MOTOR paper Equation (1):
    L(U|λ) = {λ exp(-λU)}^δ {exp(-λU)}^(1-δ)
    
    Taking log:
    log L = δ * (log(λ) - λU) + (1-δ) * (-λU)
          = δ * log(λ) - λU
    
    Negative log-likelihood (to minimize):
    -log L = -δ * log(λ) + λU
           = λU - δ * log(λ)
    
    Args:
        log_hazards: (batch_size, num_pieces) - log(λ_p)
        U: (batch_size, num_pieces) - time in each piece
        delta: (batch_size, num_pieces) - event indicator for each piece
        eps: Small constant for numerical stability
        
    Returns:
        loss: Scalar negative log-likelihood
    """
    # λ = exp(log_hazards)
    hazards = torch.exp(log_hazards)
    
    # Clamp for numerical stability
    hazards = torch.clamp(hazards, min=eps, max=1e6)
    
    # Negative log-likelihood per piece: λU - δ * log(λ)
    # = hazards * U - delta * log_hazards
    nll_per_piece = hazards * U - delta * log_hazards
    
    # Sum over pieces, mean over batch
    nll = nll_per_piece.sum(dim=1).mean()
    
    return nll


class LLMMotorModel(nn.Module):
    """
    LLM wrapper with MOTOR piecewise exponential head for time-to-event prediction.
    
    Uses the EOS token representation from the LLM as input to the piecewise
    exponential head for survival analysis.
    
    Args:
        base_model: The pretrained LLM (with or without LoRA adapters)
        hidden_size: Hidden dimension of the LLM
        num_pieces: Number of time pieces for piecewise exponential
        piece_boundaries: List of (start, end) tuples in days
        intermediate_dim: Intermediate dimension for low-rank factorization
        freeze_base: Whether to freeze the base LLM parameters
        trainable_param_keywords: Optional substrings for parameters that should remain trainable
        tokenizer: Tokenizer for debugging
    """
    
    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_pieces: int = 6,
        piece_boundaries: Optional[List[Tuple[float, float]]] = None,
        intermediate_dim: int = 64,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[List[str]] = None,
        tokenizer=None
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.trainable_param_keywords = trainable_param_keywords or []
        
        # Enable gradient checkpointing if available
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            print("  - Enabling gradient checkpointing for base model")
            self.base_model.gradient_checkpointing_enable()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("  - Froze all base model parameters")
        
        # Re-enable specific parameters if keywords provided
        if self.trainable_param_keywords:
            reenabled = 0
            for name, param in self.base_model.named_parameters():
                if any(keyword in name for keyword in self.trainable_param_keywords):
                    param.requires_grad = True
                    reenabled += 1
            print(f"  - Re-enabled {reenabled} parameters matching: {self.trainable_param_keywords}")
        
        # Piecewise exponential head
        self.tte_head = PiecewiseExponentialHead(
            hidden_size=hidden_size,
            num_pieces=num_pieces,
            piece_boundaries=piece_boundaries,
            intermediate_dim=intermediate_dim
        )
        print(f"  - Added MOTOR piecewise exponential head: {hidden_size} -> {num_pieces} pieces")
        print(f"  - Piece boundaries (days): {self.tte_head.piece_boundaries}")
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing enable to the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Helper often needed by Trainer to verify model compatibility."""
        return self.base_model.get_input_embeddings()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_times: Optional[torch.Tensor] = None,
        event_indicators: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # For compatibility, not used
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MOTOR time-to-event prediction.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            event_times: (batch_size,) time to event or censoring in days
            event_indicators: (batch_size,) 1 if event occurred, 0 if censored
            labels: Ignored, for compatibility with HuggingFace Trainer
            
        Returns:
            Dict with:
                - loss: scalar loss (if event data provided)
                - log_hazards: (batch_size, num_pieces) log-hazards per piece
                - survival_probs: (batch_size, num_pieces) survival probability at piece end
        """
        # Get backbone (handle Unsloth/PEFT wrappers)
        backbone = getattr(self.base_model, "model", self.base_model)
        
        # Get hidden states from the LLM
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract the last layer's hidden states
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Get the EOS token (last non-padding token) hidden state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.size(0)
        patient_repr = hidden_states[range(batch_size), sequence_lengths]
        
        # Compute log-hazards through piecewise exponential head
        log_hazards = self.tte_head(patient_repr)  # (batch_size, num_pieces)
        
        # Compute survival probabilities at end of each piece
        # S(t) = exp(-cumsum(λ_p * duration_p))
        hazards = torch.exp(log_hazards)
        piece_durations = torch.tensor(
            [end - start if end != float('inf') else 365 
             for start, end in self.tte_head.piece_boundaries],
            device=log_hazards.device,
            dtype=log_hazards.dtype
        )
        cumulative_hazard = torch.cumsum(hazards * piece_durations, dim=1)
        survival_probs = torch.exp(-cumulative_hazard)
        
        # Compute loss if event data provided
        loss = None
        if event_times is not None and event_indicators is not None:
            U, delta = self.tte_head.compute_time_in_pieces(event_times, event_indicators)
            loss = piecewise_exponential_loss(log_hazards, U, delta)
        
        return {
            'loss': loss,
            'log_hazards': log_hazards,
            'survival_probs': survival_probs,
            'logits': log_hazards,  # For compatibility with HuggingFace Trainer
        }
    
    def predict_survival(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict survival probabilities for given inputs.
        
        Returns survival probability at the end of each time piece.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        return {
            'survival_probs': outputs['survival_probs'],
            'log_hazards': outputs['log_hazards'],
        }
    
    def print_trainable_parameters(self):
        """Print the number of trainable vs total parameters."""
        trainable_params = 0
        all_params = 0
        print("\nTrainable parameter groups:")
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                print(f"  - {name}: {param.numel():,} params")
                trainable_params += param.numel()
        
        print(f"\n{'='*80}")
        print("Model Parameters:")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - Total params: {all_params:,}")
        print(f"  - Trainable %: {100 * trainable_params / all_params:.2f}%")
        print(f"{'='*80}\n")


def compute_c_index(
    log_hazards: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray
) -> float:
    """
    Compute Harrell's concordance index (C-index).
    
    Measures the fraction of pairs where the model correctly ranks
    the higher-risk patient (who had event earlier) with higher risk score.
    
    Args:
        log_hazards: (n_samples,) or (n_samples, num_pieces) - risk scores
                     If 2D, uses sum of log-hazards as risk score
        event_times: (n_samples,) - time to event or censoring
        event_indicators: (n_samples,) - 1 if event occurred, 0 if censored
        
    Returns:
        c_index: Concordance index (0.5 = random, 1.0 = perfect)
    """
    # If 2D, sum log-hazards as overall risk score
    if len(log_hazards.shape) == 2:
        risk_scores = log_hazards.sum(axis=1)
    else:
        risk_scores = log_hazards
    
    n = len(risk_scores)
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where at least one had an event
            # and we can determine ordering
            if event_indicators[i] == 0 and event_indicators[j] == 0:
                continue
            
            # Determine which patient had event first (or was censored later)
            if event_indicators[i] == 1 and event_indicators[j] == 1:
                # Both had events - compare times
                if event_times[i] < event_times[j]:
                    # i had event first, should have higher risk
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
                elif event_times[i] > event_times[j]:
                    # j had event first, should have higher risk
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied_risk += 1
                # If tied times, skip
                
            elif event_indicators[i] == 1 and event_indicators[j] == 0:
                # i had event, j was censored
                if event_times[i] < event_times[j]:
                    # i had event before j was censored - i should have higher risk
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied_risk += 1
                # If j was censored before i's event, can't compare
                
            elif event_indicators[i] == 0 and event_indicators[j] == 1:
                # j had event, i was censored
                if event_times[j] < event_times[i]:
                    # j had event before i was censored - j should have higher risk
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied_risk += 1
    
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5  # No comparable pairs
    
    # C-index = (concordant + 0.5 * tied) / total
    c_index = (concordant + 0.5 * tied_risk) / total
    return c_index


def compute_survival_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute survival analysis metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
                   predictions: log_hazards (batch_size, num_pieces)
                   label_ids: event_times (batch_size,)
                   We need event_indicators passed separately or encoded
    
    Returns:
        Dict of metric names to values
    """
    predictions = eval_pred.predictions
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        log_hazards = predictions[0]
    elif isinstance(predictions, dict):
        log_hazards = predictions.get('log_hazards', predictions.get('logits'))
    else:
        log_hazards = predictions
    
    # For now, return placeholder metrics
    # Full implementation requires event_indicators which need custom handling
    return {
        'mean_log_hazard': float(np.mean(log_hazards)),
        'std_log_hazard': float(np.std(log_hazards)),
    }


class MotorTrainer(Trainer):
    """
    Custom Trainer for MOTOR model that handles survival data.
    
    Extends HuggingFace Trainer to properly handle event_times and event_indicators.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with proper handling of survival data.
        """
        # Extract survival data
        event_times = inputs.pop('event_times', None)
        event_indicators = inputs.pop('event_indicators', None)
        labels = inputs.pop('labels', None)  # Remove if present, not used
        
        # Forward pass
        outputs = model(
            **inputs,
            event_times=event_times,
            event_indicators=event_indicators
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss


def run_motor_training(
    config: Dict[str, Any],
    model: nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    collate_fn
) -> Tuple[Trainer, Dict]:
    """
    Run the MOTOR time-to-event training loop.
    
    Args:
        config: Configuration dict with model, training, and data settings
        model: LLMMotorModel
        tokenizer: Tokenizer (for saving with model)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        collate_fn: Data collator function
        
    Returns:
        trainer: The trained Trainer object
        eval_results: Evaluation results dict
    """
    training_config = config['training']
    wandb_config = config.get('wandb', {})
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        run_name=wandb_config.get('run_name', 'motor-tte'),
        report_to="wandb" if wandb_config.get('enabled', False) else "none",
        
        # Training hyperparameters
        num_train_epochs=int(training_config.get('epochs', 10)),
        per_device_train_batch_size=int(training_config.get('batch_size', 8)),
        per_device_eval_batch_size=int(training_config.get('eval_batch_size', 8)),
        learning_rate=float(training_config.get('learning_rate', 1e-4)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=int(training_config.get('warmup_steps', 100)),
        
        # Gradient settings
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 1)),
        gradient_checkpointing=True,
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        
        # Precision
        fp16=bool(training_config.get('fp16', False)),
        bf16=bool(training_config.get('bf16', True)),
        
        # Logging and evaluation
        logging_steps=int(training_config.get('logging_steps', 10)),
        eval_strategy="steps",
        eval_steps=int(training_config.get('eval_steps', 100)),
        save_strategy="steps",
        save_steps=int(training_config.get('save_steps', 500)),
        save_total_limit=int(training_config.get('save_total_limit', 2)),
        
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Other
        remove_unused_columns=False,
    )
    
    # Create trainer
    print("\nInitializing MOTOR Trainer...")
    callbacks = []
    early_stopping_patience = training_config.get('early_stopping_patience', None)
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=training_config.get('early_stopping_threshold', 0.0)
        ))
        print(f"  - Early stopping enabled with patience={early_stopping_patience}")
    
    trainer = MotorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_survival_metrics,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting MOTOR time-to-event training...")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    print(f"  - Model saved to: {final_model_path}")
    
    # Run final evaluation
    print("\nRunning final evaluation on validation set...")
    eval_results = trainer.evaluate()
    print("\nFinal Validation Results:")
    print("=" * 80)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 80)
    
    # Compute C-index on validation set
    print("\nComputing C-index on validation set...")
    val_predictions = trainer.predict(val_dataset)
    log_hazards = val_predictions.predictions
    if isinstance(log_hazards, tuple):
        log_hazards = log_hazards[0]
    
    # Note: We need event_times and event_indicators from the dataset
    # This requires iterating through the dataset
    event_times_list = []
    event_indicators_list = []
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        if sample is not None:
            event_times_list.append(sample.get('time_to_event', 0))
            event_indicators_list.append(sample.get('event_indicator', 0))
    
    if event_times_list:
        event_times_np = np.array(event_times_list)
        event_indicators_np = np.array(event_indicators_list)
        
        c_index = compute_c_index(log_hazards, event_times_np, event_indicators_np)
        print(f"\nValidation C-index: {c_index:.4f}")
        eval_results['c_index'] = c_index
    
    # Test set evaluation if provided
    if test_dataset is not None:
        print("\nRunning evaluation on test set...")
        test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print("\nFinal Test Results:")
        print("=" * 80)
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)
    
    return trainer, eval_results
