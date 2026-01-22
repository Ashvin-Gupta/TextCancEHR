import torch
import torch.nn as nn
from typing import Optional

from src.pipelines.shared.blocks.mamba_block import MambaResidualBlock, RMSNorm
from src.pipelines.shared.base_models import BaseNightingaleModel

class MambaDecoder(BaseNightingaleModel):
    """
    Implementation of a Mamba-based decoder model for the Nightingale architecture.

    This model replaces the transformer's attention mechanism with Mamba's selective
    state space model, which can efficiently handle long sequences while maintaining
    compatibility with the existing Nightingale training and evaluation pipeline.

    The model follows the same interface as the transformer decoder, accepting the same
    input format and producing the same output format, making it a drop-in replacement
    for sequence modeling tasks.

    Args:
        vocab_size (int): The size of the vocabulary
        model_dim (int): The dimension of the model (d_model)
        n_layers (int): The number of Mamba layers to stack
        d_state (int): SSM state expansion factor (typically 16)
        d_conv (int): Local convolution width (typically 4)
        expand (int): Block expansion factor (typically 2)
        dt_rank (str or int): Rank of Δ projection. 'auto' sets it to ceil(d_model/16)
        dropout (float): Dropout rate for regularization
        context_length (int): Maximum sequence length the model can handle
        bias (bool): Whether to use bias in linear layers
        conv_bias (bool): Whether to use bias in convolution layers
        use_fast_path (bool): Whether to use CUDA optimizations when available
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)

        # Extract configuration
        self.vocab_size = model_config["vocab_size"]
        self.model_dim = model_config["model_dim"]
        self.n_layers = model_config["n_layers"]
        self.context_length = model_config["context_length"]
        self.dropout = model_config.get("dropout", 0.0)

        # Mamba-specific parameters with sensible defaults
        self.d_state = model_config.get("d_state", 16)
        self.d_conv = model_config.get("d_conv", 4)
        self.expand = model_config.get("expand", 2)
        self.dt_rank = model_config.get("dt_rank", "auto")
        self.bias = model_config.get("bias", False)
        self.conv_bias = model_config.get("conv_bias", True)
        self.use_fast_path = model_config.get("use_fast_path", True)

        # Token embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(self.dropout)

        # Stack of Mamba layers (using proper residual blocks)
        self.layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=self.model_dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    dt_rank=self.dt_rank,
                    conv_bias=self.conv_bias,
                    bias=self.bias,
                    use_fast_path=self.use_fast_path,
                    layer_idx=i,
                    dropout=self.dropout,
                )
                for i in range(self.n_layers)
            ]
        )

        # Final normalization before output (matches RMSNorm used in blocks)
        self.norm = RMSNorm(self.model_dim)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def required_config_keys(self) -> set[str]:
        """
        Returns the required configuration keys for this model.

        Returns:
            set[str]: Set of required configuration key names
        """
        return {"vocab_size", "model_dim", "n_layers", "context_length"}

    def required_input_keys(self) -> set[str]:
        """Returns the required input keys for this model."""
        # return {"ehr.input_token_ids"}
        return {"ehr.input_token_ids", "ehr.input_padding_mask"}

    def _init_weights(self) -> None:
        """
        Initialize model weights using appropriate initialization schemes.

        Uses normal initialization for embeddings following common practices in language modeling.
        Mamba blocks handle their own internal initialization.
        """
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Initialize output projection
        nn.init.normal_(self.lm_head.weight, std=0.02)

        # RMSNorm parameters are initialized by default to ones

        # Mamba blocks handle their own initialization internally

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the Mamba decoder model.

        Args:
            x (dict): Input dictionary containing:
                - ehr.input_token_ids (torch.Tensor): Token sequence of shape (batch_size, sequence_length)
                - ehr.input_padding_mask (torch.Tensor): Binary mask of shape (batch_size, sequence_length)
                  where True/1 indicates real tokens and False/0 indicates padding tokens

        Returns:
            torch.Tensor: Output logits of shape (batch_size, sequence_length, vocab_size)
                representing unnormalized probabilities for next token prediction
        """
        # Validate inputs
        self.validate_input(x)

        # Extract inputs
        ehr_inputs = x["ehr"]
        input_token_ids = ehr_inputs["input_token_ids"]  # (batch_size, seq_len)
        padding_mask = ehr_inputs["input_padding_mask"]  # (batch_size, seq_len)

        batch_size, seq_len = input_token_ids.shape

        # Ensure padding mask is on the correct device and dtype
        padding_mask = padding_mask.to(device=input_token_ids.device, dtype=torch.bool)

        # Token embeddings
        hidden_states = self.embedding(input_token_ids)  # (batch_size, seq_len, model_dim)

        # Apply embedding dropout
        hidden_states = self.emb_dropout(hidden_states)

        # Zero out embeddings for padded positions (more efficient than repeated masking)
        padding_mask_expanded = padding_mask.unsqueeze(-1)
        hidden_states = hidden_states * padding_mask_expanded.float()

        # Pass through Mamba layers
        # Note: Mamba's SSM naturally handles variable-length sequences through its convolution
        # and state space operations. The padding mask ensures padded positions don't contribute.
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=padding_mask)

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Zero out outputs for padded positions
        hidden_states = hidden_states * padding_mask_expanded.float()

        # Project to vocabulary size
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

        # Zero out logits for padded positions to ensure they don't affect loss/sampling
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(logits)
        logits = logits * padding_mask_expanded.float()

        return logits

    def generate_step(self, input_ids: torch.Tensor, cache: Optional[dict] = None, **kwargs) -> tuple[torch.Tensor, dict]:
        """
        Generate a single token for autoregressive generation.

        This method is used during inference for efficient token-by-token generation
        while maintaining the internal state of Mamba blocks.

        Args:
            input_ids (torch.Tensor): Input token of shape (batch_size, 1)
            cache (dict, optional): Cache containing conv_state and ssm_state for each layer
            **kwargs: Additional generation parameters (unused)

        Returns:
            tuple[torch.Tensor, dict]:
                - logits: Output logits of shape (batch_size, 1, vocab_size)
                - cache: Updated cache for next generation step
        """
        if cache is None:
            # Initialize cache for all layers
            batch_size = input_ids.shape[0]
            cache = {}
            for i, layer in enumerate(self.layers):
                layer_cache = layer.allocate_inference_cache(batch_size, max_seqlen=1)
                cache[f"layer_{i}"] = layer_cache

        # Embedding
        hidden_states = self.embedding(input_ids)  # (batch_size, 1, model_dim)
        hidden_states = self.emb_dropout(hidden_states)

        # Pass through each layer with state updates
        for i, layer in enumerate(self.layers):
            layer_cache = cache[f"layer_{i}"]
            # Use step method for single token processing
            if hasattr(layer, 'step'):
                hidden_states_single = hidden_states.squeeze(1)  # Remove seq_len dimension for step
                hidden_states_single, layer_cache = layer.step(hidden_states_single, layer_cache)
                hidden_states = hidden_states_single.unsqueeze(1)  # Add seq_len dimension back
                cache[f"layer_{i}"] = layer_cache  # Update cache
            else:
                # Fallback to regular forward if step is not available
                hidden_states = layer(hidden_states)

        # Final processing
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, cache

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding (bool): If True, exclude embedding parameters from count

        Returns:
            int: Number of parameters
        """
        total_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract embedding parameters
            embedding_params = self.embedding.weight.numel()
            total_params -= embedding_params

        return total_params

    def get_model_info(self) -> dict:
        """
        Get comprehensive information about the model architecture.

        Returns:
            dict: Dictionary containing model architecture information
        """
        total_params = self.get_num_params(non_embedding=False)
        non_embedding_params = self.get_num_params(non_embedding=True)

        return {
            "model_type": "mamba",
            "vocab_size": self.vocab_size,
            "model_dim": self.model_dim,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "context_length": self.context_length,
            "total_parameters": total_params,
            "non_embedding_parameters": non_embedding_params,
            "parameters_M": round(total_params / 1e6, 2),
            "fast_path_enabled": self.layers[0].use_fast_path if self.layers else False,
        }


if __name__ == "__main__":
    # Simple test of the Mamba model
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    model_dim = 64

    # Create model configuration
    model_config = {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "n_layers": 4,
        "context_length": 128,
        "dropout": 0.1,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_fast_path": False
    }

    # Create test input
    input_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    x = {
        "ehr": {
            "input_token_ids": input_token_ids,
            "input_padding_mask": padding_mask,
        }
    }

    # Create and test the model
    model = MambaDecoder(model_config)

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print("=== Mamba Model Test ===")
    print(f"Input shape: {input_token_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model info: {model.get_model_info()}")

    # Test generation step
    print("\n=== Generation Test ===")
    single_token = torch.randint(0, vocab_size, (batch_size, 1))
    logits_gen, cache = model.generate_step(single_token)
    print(f"Generation input shape: {single_token.shape}")
    print(f"Generation output shape: {logits_gen.shape}")
    print(f"Cache keys: {list(cache.keys())}")

    print("\nMamba model test completed successfully!")