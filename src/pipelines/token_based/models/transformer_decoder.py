import math
import torch
from src.pipelines.shared.blocks.multihead_attention import MultiHeadAttention
from src.pipelines.shared.base_models import BaseNightingaleModel


class TransformerDecoder(BaseNightingaleModel):
    """
    Implementation of a GPT-style transformer decoder (https://arxiv.org/abs/1706.03762)

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden layer.
        n_layers (int): The number of layers in the transformer.
        dropout (float): The dropout rate.
        n_heads (int): The number of attention heads.
        context_length (int): The length of the context.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.model_dim = model_config["model_dim"]
        self.n_layers = model_config["n_layers"]
        self.context_length = model_config["context_length"]

        # Embedding matrix
        self.embedding = torch.nn.Embedding(model_config["vocab_size"], model_config["model_dim"])

        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_config["model_dim"], model_config["dropout"], model_config["context_length"])

        # Create the transformer decoder layers
        # input layer
        self.layers = torch.nn.ModuleList(
            [TransformerDecoderBlock(d_model=model_config["model_dim"], n_heads=model_config["n_heads"], dropout=model_config["dropout"])
            for _ in range(self.n_layers)]
        )

        # output projection
        self.linear = torch.nn.Linear(model_config["model_dim"], model_config["vocab_size"])

        # Initialize weights using Xavier uniform initialization
        self._init_weights()

    def required_config_keys(self) -> set[str]:
        return {"vocab_size", "model_dim", "n_layers", "dropout", "n_heads", "context_length"}

    def required_input_keys(self) -> set[str]:
        return {"ehr.input_token_ids"}

    def _init_weights(self) -> None:
        """
        Initialize weight matrices using Xavier uniform initialization.
        Leaves biases to be initialized by PyTorch.
        """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the transformer decoder model.

        Args:
            x (dict): Input dictionary, with relevant keys:
                - ehr.input_token_ids (torch.Tensor): The input token sequence of shape (batch_size, sequence_length).

        Returns:
            y (torch.Tensor): The output logits of shape (batch_size, sequence_length, vocab_size). The logits are the
                unnormalized probabilities of the next token in the sequence.
        """

        # validate input
        self.validate_input(x)

        input_token_ids = x["ehr"]["input_token_ids"] # (batch_size, sequence_length)

        # embed token sequence with positional encoding
        embedded = self.embedding(input_token_ids)
        embedded = self.pos_encoding(embedded)

        # pass through transformer decoder layers sequentially
        output = embedded
        for layer in self.layers:
            output = layer(output) # (batch_size, sequence_length, model_dim)

        # pass through linear layer
        y = self.linear(output) # (batch_size, sequence_length, vocab_size)

        return y


class TransformerDecoderBlock(torch.nn.Module):
    """
    Implementation of a single transformer decoder block (https://arxiv.org/abs/1706.03762)

    Args:
        d_model (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        dim_feedforward (int): The dimension of the feedforward layer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model),
            torch.nn.Dropout(dropout),
        )
        self.resid_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single transformer decoder block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            x (torch.Tensor): The output tensor of shape (batch_size, seq_len, d_model).
        """
        # pre-LN attention
        x = x + self.attn(self.ln1(x))

        # pre-LN MLP
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))

        return x


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for transformer models.

    Args:
        d_model (int): The dimension of the model.
        dropout (float): The dropout rate.
        max_len (int): The maximum length of the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            x (torch.Tensor): Embeddings with positional encoding added of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


if __name__ == "__main__":
    # define params
    batch_size = 1
    sequence_length = 10
    vocab_size = 100
    num_heads = 1
    model_dim = num_heads * 64
    n_layers = 2
    dropout = 0.5

    # random input
    rand = torch.randint(0, vocab_size, (batch_size, sequence_length + 1))
    x = {
        "ehr": {
            "input_token_ids": rand[:, :-1]
        }
    }
    target = {
        "ehr": {
            "target_token_ids": rand[:, 1:]
        }
    }

    model_config = {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "n_heads": num_heads,
        "context_length": sequence_length,
    }

    # init model and forward pass
    model = TransformerDecoder(model_config=model_config)
    pred = model(x)

    # print loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred.view(-1, vocab_size), target["ehr"]["input_token_ids"].view(-1))
    print(f"Loss: {loss}")
