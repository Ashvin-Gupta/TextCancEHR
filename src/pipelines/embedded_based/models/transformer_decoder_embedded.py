# src/models/core_models/transformer_decoder_embedded.py

"""
Transformer Decoder model for pre-computed embeddings (e.g., from E5).

This model is designed for autoregressive next-event prediction using pre-embedded
EHR data. It uses causal (unidirectional) attention and predicts the next token ID.
"""

import math
import torch
import torch.nn as nn
from src.pipelines.shared.base_models import BaseNightingaleModel


class TransformerDecoderEmbedded(BaseNightingaleModel):
    """
    Transformer decoder that works with pre-computed embeddings for autoregressive modeling.
    
    Architecture:
        Input Embeddings (N, D) → Linear Projection → Positional Encoding →
        Transformer Decoder Blocks (Causal) → Linear Head → Token Logits
    
    Args:
        model_config (dict):
            - embedding_dim: Dimension of input embeddings (e.g., 768 for E5)
            - model_dim: Internal model dimension
            - n_layers: Number of transformer decoder layers
            - n_heads: Number of attention heads
            - dropout: Dropout rate
            - context_length: Maximum sequence length
            - vocab_size: Size of output vocabulary for next-token prediction
            - add_classification_head: Whether to add a classification head (optional)
            - num_classes: Number of classes if classification head is used
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.embedding_dim = model_config["embedding_dim"]
        self.model_dim = model_config["model_dim"]
        self.n_layers = model_config["n_layers"]
        self.context_length = model_config["context_length"]
        self.vocab_size = model_config["vocab_size"]
        self.add_classification_head = model_config.get("add_classification_head", False)
        
        # Project input embeddings to model dimension
        if self.embedding_dim != self.model_dim:
            self.input_projection = nn.Linear(self.embedding_dim, self.model_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.model_dim,
            model_config["dropout"],
            model_config["context_length"]
        )
        
        # Transformer decoder layers (causal attention)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=model_config["n_heads"],
            dim_feedforward=4 * self.model_dim,
            dropout=model_config["dropout"],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.n_layers
        )
        
        # Output head for next-token prediction
        self.lm_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.vocab_size)
        )
        
        # Optional classification head
        if self.add_classification_head:
            self.num_classes = model_config.get("num_classes", 2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.model_dim),
                nn.Linear(self.model_dim, self.num_classes)
            )
        
        # Initialize weights
        self._init_weights()

    def required_config_keys(self) -> set[str]:
        return {
            "embedding_dim", "model_dim", "n_layers",
            "dropout", "n_heads", "context_length", "vocab_size"
        }

    def required_input_keys(self) -> set[str]:
        return {"input_embeddings", "padding_mask"}

    def _init_weights(self) -> None:
        """Initialize weight matrices using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal (triangular) attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask of shape (seq_len, seq_len) where True = attend, False = mask
        """
        # Create causal mask: lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Convert to boolean: True = can attend, False = masked
        mask = mask.bool()
        return mask

    def forward(self, x: dict, return_classification: bool = False) -> torch.Tensor:
        """
        Forward pass of the transformer decoder model.

        Args:
            x (dict): Input dictionary with keys:
                - input_embeddings (torch.Tensor): Pre-computed input embeddings
                  of shape (batch_size, seq_len, embedding_dim)
                - padding_mask (torch.Tensor): Boolean mask where True = valid token
                  of shape (batch_size, seq_len)
            return_classification (bool): If True and classification head exists,
                                         return classification logits instead of LM logits

        Returns:
            torch.Tensor: 
                - If return_classification=False: Token logits of shape (batch_size, seq_len, vocab_size)
                - If return_classification=True: Classification logits of shape (batch_size, num_classes)
        """
        self.validate_input(x)
        
        input_embeddings = x["input_embeddings"]  # (B, T, D_embed)
        padding_mask = x["padding_mask"]  # (B, T) - True = valid
        batch_size, seq_len, _ = input_embeddings.shape
        device = input_embeddings.device
        
        # Project to model dimension
        embedded = self.input_projection(input_embeddings)  # (B, T, D_model)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
     
        # Create boolean causal attention mask (True = masked)
        # This prevents attention to future tokens.
        attn_mask = torch.full((seq_len, seq_len), True, device=device, dtype=torch.bool).triu(diagonal=1)
        # Create padding mask (True = padding)
        src_key_padding_mask = ~padding_mask  # Invert: True = padding
        # Pass through transformer decoder with causal mask
        output = self.transformer_decoder(
            embedded,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, T, D_model)
        # Return appropriate output
        if return_classification and self.add_classification_head:
            # Pool and classify
            mask_expanded = padding_mask.unsqueeze(-1)  # (B, T, 1)
            masked_output = output * mask_expanded
            sum_output = masked_output.sum(dim=1)  # (B, D_model)
            seq_lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = sum_output / seq_lengths  # (B, D_model)
            return self.classifier(pooled)  # (B, num_classes)
        else:
            # Next-token prediction
            logits = self.lm_head(output)  # (B, T, vocab_size)
            return logits


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Args:
        d_model (int): The dimension of the model.
        dropout (float): The dropout rate.
        max_len (int): The maximum length of the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Embeddings with positional encoding of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 100
    embedding_dim = 768
    model_dim = 256
    n_layers = 4
    n_heads = 8
    vocab_size = 1000
    num_classes = 2
    
    # Create random input
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    # Create padding mask
    seq_lengths = torch.tensor([80, 100, 60, 90])
    padding_mask = torch.arange(seq_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    
    # Test 1: Autoregressive model (no classification head)
    print("=" * 60)
    print("Test 1: Autoregressive next-token prediction")
    print("=" * 60)
    
    model_config_lm = {
        "embedding_dim": embedding_dim,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": 0.1,
        "context_length": 512,
        "vocab_size": vocab_size
    }
    
    model_lm = TransformerDecoderEmbedded(model_config_lm)
    
    x = {
        "input_embeddings": input_embeddings,
        "padding_mask": padding_mask
    }
    logits = model_lm(x)
    
    print(f"Input shape: {input_embeddings.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Test loss computation
    target_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    loss = loss_fn(logits.view(-1, vocab_size), target_token_ids.view(-1))
    print(f"LM Loss: {loss.item():.4f}")
    
    # Test 2: Model with classification head
    print("\n" + "=" * 60)
    print("Test 2: Autoregressive + Classification head")
    print("=" * 60)
    
    model_config_both = {
        "embedding_dim": embedding_dim,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": 0.1,
        "context_length": 512,
        "vocab_size": vocab_size,
        "add_classification_head": True,
        "num_classes": num_classes
    }
    
    model_both = TransformerDecoderEmbedded(model_config_both)
    
    # Get LM logits
    lm_logits = model_both(x, return_classification=False)
    print(f"LM logits shape: {lm_logits.shape}")
    
    # Get classification logits
    cls_logits = model_both(x, return_classification=True)
    print(f"Classification logits shape: {cls_logits.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes})")
    
    # Test classification loss
    labels = torch.randint(0, num_classes, (batch_size,))
    cls_loss_fn = nn.CrossEntropyLoss()
    cls_loss = cls_loss_fn(cls_logits, labels)
    print(f"Classification Loss: {cls_loss.item():.4f}")

