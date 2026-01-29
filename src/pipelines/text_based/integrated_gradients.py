# src/pipelines/text_based/integrated_gradients.py

"""
Integrated Gradients implementation for LLM classifier interpretability.

Provides token-level attribution scores using the Captum library, showing which
input tokens contribute most to cancer predictions.

Key features:
- Multiple baseline strategies (PAD, UNK, zero)
- Aggregation to event and concept levels
- Handles long sequences (up to 12k tokens)
- BFloat16 compatible
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import warnings


@dataclass
class TokenAttributions:
    """Container for token-level attribution results."""
    attributions: np.ndarray  # (seq_len,) attribution scores
    tokens: List[str]  # Decoded tokens
    token_ids: np.ndarray  # (seq_len,) token IDs
    baseline_prediction: float  # P(cancer) for baseline
    input_prediction: float  # P(cancer) for actual input
    completeness_score: float  # sum(attributions) vs (input - baseline)
    text: str  # Original text


@dataclass
class EventAttributions:
    """Container for event-level aggregated attributions."""
    event_attributions: List[Tuple[str, float]]  # (event_text, attribution)
    event_positions: List[Tuple[int, int]]  # (start_token, end_token)
    aggregation_method: str  # 'sum' or 'mean'


@dataclass
class ConceptAttributions:
    """Container for medical concept attributions."""
    concept_attributions: List[Tuple[str, str, float]]  # (concept_text, code, attribution)
    concept_positions: List[Tuple[int, int]]  # (start_token, end_token)
    concept_categories: List[str]  # Category for each concept


class CaptumModelWrapper(nn.Module):
    """
    Wrapper for LLMClassifier to work with Captum's IntegratedGradients.
    
    Captum requires:
    - Forward method that takes input_ids
    - Returns scalar output (cancer logit)
    - Handles attention masks internally
    """
    
    def __init__(self, model, attention_mask: torch.Tensor):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Captum.
        
        Args:
            input_ids: (batch_size, seq_len) or (seq_len,)
        
        Returns:
            Cancer logit: (batch_size,) or scalar
        """
        # Ensure batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Run model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=self.attention_mask
        )
        
        logits = outputs['logits']  # (batch_size, 2)
        
        # Return cancer logit (class 1)
        return logits[:, 1]


def create_baseline(
    input_ids: torch.Tensor,
    tokenizer,
    strategy: str = 'pad'
) -> torch.Tensor:
    """
    Create a baseline for Integrated Gradients.
    
    Args:
        input_ids: (seq_len,) or (1, seq_len)
        tokenizer: Tokenizer with special tokens
        strategy: 'pad', 'unk', 'mask', or 'zero'
    
    Returns:
        Baseline tensor of same shape as input_ids
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    if strategy == 'pad':
        # All tokens = PAD token
        baseline = torch.full(
            (batch_size, seq_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device
        )
    
    elif strategy == 'unk':
        # All tokens = UNK token
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.pad_token_id
        baseline = torch.full(
            (batch_size, seq_len),
            unk_id,
            dtype=torch.long,
            device=device
        )
    
    elif strategy == 'mask':
        # All tokens = MASK token (if available)
        mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else tokenizer.pad_token_id
        baseline = torch.full(
            (batch_size, seq_len),
            mask_id,
            dtype=torch.long,
            device=device
        )
    
    elif strategy == 'zero':
        # Zero token IDs (will be embedded as zero vectors if using embedding-level IG)
        baseline = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
    
    return baseline


def compute_integrated_gradients(
    model,
    tokenizer,
    text: str,
    baseline_strategy: str = 'pad',
    n_steps: int = 50,
    max_length: int = 12000,
    device: str = "cuda",
    use_embedding_layer: bool = True
) -> TokenAttributions:
    """
    Compute Integrated Gradients attributions for a text input.
    
    Args:
        model: LLMClassifier model
        tokenizer: Tokenizer
        text: Input text
        baseline_strategy: 'pad', 'unk', 'mask', or 'zero'
        n_steps: Number of steps for IG approximation
        max_length: Maximum sequence length
        device: Device to run on
        use_embedding_layer: If True, compute IG at embedding level (more accurate but slower)
    
    Returns:
        TokenAttributions with per-token scores
    """
    model.eval()
    
    # Convert model to float32 for gradient computation (required for some operations)
    # Store original dtype to restore later
    original_dtype = next(model.parameters()).dtype
    if original_dtype == torch.bfloat16:
        print(f"  Converting model from {original_dtype} to float32 for gradient computation...")
        model.to(torch.float32)
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    # Ensure input_ids and attention_mask are Long type (required for embeddings)
    input_ids = encoding["input_ids"].to(device).long()
    attention_mask = encoding["attention_mask"].to(device).long()
    
    seq_len = input_ids.shape[1]
    
    # Create baseline
    baseline_ids = create_baseline(input_ids, tokenizer, baseline_strategy)
    
    # Create model wrapper
    wrapped_model = CaptumModelWrapper(model, attention_mask)
    
    # Initialize Integrated Gradients
    if use_embedding_layer:
        # Attribute at embedding level (more accurate)
        # Get embedding layer
        embedding_layer = model.base_model.get_input_embeddings()
        ig = LayerIntegratedGradients(wrapped_model, embedding_layer)
        
        # Compute attributions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            attributions, delta = ig.attribute(
                input_ids,
                baselines=baseline_ids,
                n_steps=n_steps,
                return_convergence_delta=True
            )
        
        # Sum over embedding dimension to get per-token scores
        # attributions shape: (1, seq_len, embed_dim)
        attributions = attributions.sum(dim=-1)  # (1, seq_len)
    
    else:
        # Attribute at input level (faster)
        ig = IntegratedGradients(wrapped_model)
        
        # Compute attributions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            attributions, delta = ig.attribute(
                input_ids,
                baselines=baseline_ids,
                n_steps=n_steps,
                return_convergence_delta=True,
                internal_batch_size=1
            )
        # attributions shape: (1, seq_len)
    
    # Convert to numpy
    attributions = attributions[0].cpu().detach().numpy()  # (seq_len,)
    
    # Get predictions for completeness check
    with torch.no_grad():
        # Baseline prediction
        baseline_outputs = model(input_ids=baseline_ids, attention_mask=attention_mask)
        baseline_logits = baseline_outputs['logits'][0].cpu().numpy()
        baseline_prob = 1 / (1 + np.exp(-(baseline_logits[1] - baseline_logits[0])))
        
        # Input prediction
        input_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        input_logits = input_outputs['logits'][0].cpu().numpy()
        input_prob = 1 / (1 + np.exp(-(input_logits[1] - input_logits[0])))
    
    # Completeness check: sum(attributions) should ≈ (input_logit - baseline_logit)
    attribution_sum = np.sum(attributions)
    actual_diff = input_logits[1] - baseline_logits[1]
    completeness_score = abs(attribution_sum - actual_diff) / (abs(actual_diff) + 1e-8)
    
    # Decode tokens
    token_ids_np = input_ids[0].cpu().numpy()
    tokens = [tokenizer.decode([tid]) for tid in token_ids_np]
    
    # Restore original dtype (in-place conversion)
    if original_dtype == torch.bfloat16:
        print(f"  Restoring model to {original_dtype}...")
        model.to(original_dtype)
    
    return TokenAttributions(
        attributions=attributions,
        tokens=tokens,
        token_ids=token_ids_np,
        baseline_prediction=float(baseline_prob),
        input_prediction=float(input_prob),
        completeness_score=float(completeness_score),
        text=text
    )


def aggregate_attributions_to_events(
    token_attrs: TokenAttributions,
    delimiter: str = ';',
    aggregation_method: str = 'sum'
) -> EventAttributions:
    """
    Aggregate token-level attributions to event level.
    
    Events are defined as text segments between delimiters.
    
    Args:
        token_attrs: TokenAttributions from compute_integrated_gradients
        delimiter: Event delimiter (default ';' for your EHR format)
        aggregation_method: 'sum' or 'mean'
    
    Returns:
        EventAttributions with event-level scores
    """
    tokens = token_attrs.tokens
    attributions = token_attrs.attributions
    
    # Find event boundaries
    events = []
    event_positions = []
    current_event_tokens = []
    current_event_attrs = []
    current_start = 0
    
    for i, token in enumerate(tokens):
        current_event_tokens.append(token)
        current_event_attrs.append(attributions[i])
        
        # Check if this token contains the delimiter
        if delimiter in token or i == len(tokens) - 1:
            # End of event
            event_text = "".join(current_event_tokens).strip()
            
            if aggregation_method == 'sum':
                event_attr = np.sum(current_event_attrs)
            elif aggregation_method == 'mean':
                event_attr = np.mean(current_event_attrs)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            if event_text:  # Non-empty event
                events.append((event_text, float(event_attr)))
                event_positions.append((current_start, i + 1))
            
            # Reset for next event
            current_event_tokens = []
            current_event_attrs = []
            current_start = i + 1
    
    return EventAttributions(
        event_attributions=events,
        event_positions=event_positions,
        aggregation_method=aggregation_method
    )


def aggregate_attributions_to_concepts(
    token_attrs: TokenAttributions,
    medical_terms: Dict[str, str],
    aggregation_method: str = 'mean'
) -> ConceptAttributions:
    """
    Aggregate token-level attributions to medical concept level.
    
    Args:
        token_attrs: TokenAttributions from compute_integrated_gradients
        medical_terms: Dictionary mapping term text (lowercase) to code
        aggregation_method: 'sum' or 'mean'
    
    Returns:
        ConceptAttributions with concept-level scores
    """
    from src.pipelines.text_based.feature_ablation_analysis import categorize_medical_term
    
    text = token_attrs.text
    text_lower = text.lower()
    tokens = token_attrs.tokens
    attributions = token_attrs.attributions
    
    # Build position-to-token mapping
    # Approximate token positions in the original text
    token_positions = []
    current_pos = 0
    for token in tokens:
        token_positions.append(current_pos)
        current_pos += len(token)
    
    concept_results = []
    concept_positions_tokens = []
    concept_categories = []
    
    # Find each medical term in the text
    for term_text, term_code in medical_terms.items():
        # Find all occurrences
        start_pos = 0
        while True:
            pos = text_lower.find(term_text, start_pos)
            if pos == -1:
                break
            
            end_pos = pos + len(term_text)
            
            # Find which tokens overlap with this concept
            overlapping_tokens = []
            overlapping_attrs = []
            
            for i, token_pos in enumerate(token_positions):
                token_end = token_pos + len(tokens[i])
                
                # Check if token overlaps with concept span
                if token_pos < end_pos and token_end > pos:
                    overlapping_tokens.append(i)
                    overlapping_attrs.append(attributions[i])
            
            if overlapping_attrs:
                # Aggregate attributions
                if aggregation_method == 'sum':
                    concept_attr = np.sum(overlapping_attrs)
                elif aggregation_method == 'mean':
                    concept_attr = np.mean(overlapping_attrs)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
                
                # Get actual text
                actual_text = text[pos:end_pos]
                
                # Categorize
                category = categorize_medical_term(term_text)
                
                concept_results.append((actual_text, term_code, float(concept_attr)))
                concept_positions_tokens.append((min(overlapping_tokens), max(overlapping_tokens) + 1))
                concept_categories.append(category)
            
            start_pos = end_pos
    
    return ConceptAttributions(
        concept_attributions=concept_results,
        concept_positions=concept_positions_tokens,
        concept_categories=concept_categories
    )


def visualize_token_attributions(
    token_attrs: TokenAttributions,
    output_path: str,
    top_k: int = 100,
    max_display_tokens: int = 200
) -> None:
    """
    Create a simple text-based visualization of token attributions.
    
    For HTML heatmap, consider using a separate visualization tool.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Get top positive and negative attributions
    sorted_indices = np.argsort(np.abs(token_attrs.attributions))[::-1]
    top_indices = sorted_indices[:top_k]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Top positive attributions (support cancer prediction)
    positive_mask = token_attrs.attributions[top_indices] > 0
    positive_indices = top_indices[positive_mask][:25]
    
    if len(positive_indices) > 0:
        tokens_pos = [f"[{i}] {token_attrs.tokens[i][:15]}" for i in positive_indices]
        attrs_pos = [token_attrs.attributions[i] for i in positive_indices]
        
        ax1.barh(range(len(tokens_pos)), attrs_pos, color='crimson', edgecolor='black', alpha=0.7)
        ax1.set_yticks(range(len(tokens_pos)))
        ax1.set_yticklabels(tokens_pos, fontsize=9)
        ax1.set_xlabel('Attribution Score', fontsize=12)
        ax1.set_title('Top 25 Tokens Supporting Cancer Prediction (Positive Attribution)', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()
    
    # Plot 2: Top negative attributions (support control prediction)
    negative_mask = token_attrs.attributions[top_indices] < 0
    negative_indices = top_indices[negative_mask][:25]
    
    if len(negative_indices) > 0:
        tokens_neg = [f"[{i}] {token_attrs.tokens[i][:15]}" for i in negative_indices]
        attrs_neg = [token_attrs.attributions[i] for i in negative_indices]
        
        ax2.barh(range(len(tokens_neg)), attrs_neg, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_yticks(range(len(tokens_neg)))
        ax2.set_yticklabels(tokens_neg, fontsize=9)
        ax2.set_xlabel('Attribution Score', fontsize=12)
        ax2.set_title('Top 25 Tokens Supporting Control Prediction (Negative Attribution)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

