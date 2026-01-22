import torch
from src.pipelines.shared.base_models import BaseNightingaleModel


class LSTM(BaseNightingaleModel):
    def __init__(self,model_config: dict):
        super().__init__(model_config)
        self.embedding = torch.nn.Embedding(model_config["vocab_size"], model_config["embedding_dim"])
        self.lstm = torch.nn.LSTM(
            model_config["embedding_dim"], model_config["hidden_dim"], model_config["n_layers"], dropout=model_config["dropout"], batch_first=True
        )
        self.fc = torch.nn.Linear(model_config["hidden_dim"], model_config["vocab_size"])

    def required_config_keys(self) -> set[str]:
        return {"vocab_size", "embedding_dim", "hidden_dim", "n_layers", "dropout"}

    def required_input_keys(self) -> set[str]:
        return {"ehr.input_token_ids"}

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

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

        # embed token sequence
        embedded = self.embedding(input_token_ids)

        # pass through LSTM
        output, (hidden, cell) = self.lstm(embedded) # (batch_size, sequence_length, hidden_dim)

        # pass through linear layer
        y = self.fc(output) # (batch_size, sequence_length, vocab_size)

        return y


if __name__ == "__main__":
    # define params
    batch_size = 3
    sequence_length = 10
    vocab_size = 100
    embedding_dim = 100
    hidden_dim = 100

    # random input
    x = {
        "ehr": {
            "input_token_ids": torch.randint(0, vocab_size, (batch_size, sequence_length))
        }
    }

    # init model and forward pass
    model = LSTM(model_config={
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_layers": 2,
        "dropout": 0.5,
    })
    y = model(x)
    print(f"Output: {y.shape}")