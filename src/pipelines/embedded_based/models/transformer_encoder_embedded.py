# src/models/core_models/transformer_encoder_embedded.py

"""
Transformer Encoder model for pre-computed embeddings (e.g., from E5).

This model is designed for classification tasks using pre-embedded EHR data.
It uses bidirectional attention (no causal mask) and pools the sequence
representation for classification.
"""

import math
import torch
import torch.nn as nn
from src.pipelines.shared.base_models import BaseNightingaleModel


class TransformerEncoderEmbedded(BaseNightingaleModel):
    """
    Transformer encoder that works with pre-computed embeddings.
    
    Architecture:
        Input Embeddings (N, 768) → Linear Projection → Positional Encoding →
        Transformer Encoder Blocks → Mean Pooling → Classification Head
    
    Args:
        model_config (dict):
            - embedding_dim: Dimension of input embeddings (e.g., 768 for E5)
            - model_dim: Internal model dimension
            - n_layers: Number of transformer encoder layers
            - n_heads: Number of attention heads
            - dropout: Dropout rate
            - context_length: Maximum sequence length
            - num_classes: Number of output classes
            - pooling: Pooling strategy ('mean', 'cls', 'max') - default 'mean'
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.embedding_dim = model_config["embedding_dim"]
        self.model_dim = model_config["model_dim"]
        self.n_layers = model_config["n_layers"]
        self.context_length = model_config["context_length"]
        self.num_classes = model_config["num_classes"]
        self.pooling = model_config.get("pooling", "mean")
        
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
        
        # Transformer encoder layers (bidirectional attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=model_config["n_heads"],
            dim_feedforward=4 * self.model_dim,
            dropout=model_config["dropout"],
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def required_config_keys(self) -> set[str]:
        return {
            "embedding_dim", "model_dim", "n_layers",
            "dropout", "n_heads", "context_length", "num_classes"
        }

    def required_input_keys(self) -> set[str]:
        return {"embeddings", "padding_mask"}

    def _init_weights(self) -> None:
        """Initialize weight matrices using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the transformer encoder model.

        Args:
            x (dict): Input dictionary with keys:
                - embeddings (torch.Tensor): Pre-computed embeddings
                  of shape (batch_size, seq_len, embedding_dim)
                - padding_mask (torch.Tensor): Boolean mask where True = valid token
                  of shape (batch_size, seq_len)

        Returns:
            logits (torch.Tensor): Classification logits of shape (batch_size, num_classes)
        """
        self.validate_input(x)
        
        embeddings = x["embeddings"]  # (B, T, 768)
        padding_mask = x["padding_mask"]  # (B, T) - True = valid
        
        # Project to model dimension
        embedded = self.input_projection(embeddings)  # (B, T, model_dim)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Transformer encoder expects padding mask as (B, T) with True = padding (opposite!)
        # So we need to invert our mask
        src_key_padding_mask = ~padding_mask  # True = padding
        
        # Pass through transformer encoder
        output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )  # (B, T, model_dim)
        
        # Pool over sequence dimension
        if self.pooling == 'mean':
            # Mean pooling with mask
            mask_expanded = padding_mask.unsqueeze(-1)  # (B, T, 1)
            masked_output = output * mask_expanded  # Zero out padding
            sum_output = masked_output.sum(dim=1)  # (B, model_dim)
            seq_lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled = sum_output / seq_lengths  # (B, model_dim)
        elif self.pooling == 'cls':
            # Use first token (assumes CLS token at position 0)
            pooled = output[:, 0, :]  # (B, model_dim)
        elif self.pooling == 'max':
            # Max pooling with mask
            mask_expanded = padding_mask.unsqueeze(-1)  # (B, T, 1)
            masked_output = output.masked_fill(~mask_expanded, float('-inf'))
            pooled = masked_output.max(dim=1)[0]  # (B, model_dim)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Classify
        logits = self.classifier(pooled)  # (B, num_classes)
        
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
    num_classes = 2
    
    # Create random input
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    # Create padding mask (first patient has 80 tokens, second has 100, etc.)
    seq_lengths = torch.tensor([80, 100, 60, 90])
    padding_mask = torch.arange(seq_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
    
    # Create model
    model_config = {
        "embedding_dim": embedding_dim,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": 0.1,
        "context_length": 512,
        "num_classes": num_classes,
        "pooling": "mean"
    }
    
    model = TransformerEncoderEmbedded(model_config)
    
    # Forward pass
    x = {
        "embeddings": embeddings,
        "padding_mask": padding_mask
    }
    logits = model(x)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {num_classes})")
    
    # Test loss computation
    labels = torch.randint(0, num_classes, (batch_size,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"Loss: {loss.item():.4f}")

