import torch

class BaseNightingaleModel(torch.nn.Module):
    """
    Base class for all Nightingale models.
    """
    def __init__(self, model_config: dict):
        super().__init__()

        # validate config
        self.validate_config(model_config)

        # store config
        self.model_config = model_config

    @staticmethod
    def required_config_keys() -> set[str]:
        """
        Returns the required keys for the model configuration.
        """
        raise NotImplementedError("Required keys not implemented")

    def validate_config(self, model_config: dict) -> None:
        """
        Validates the model configuration.
        """
        for key in self.required_config_keys():
            if key not in model_config:
                raise ValueError(f"Missing required config key: {key}")

    @staticmethod
    def required_input_keys() -> set[str]:
        """
        Returns the required keys for the model input.
        """
        raise NotImplementedError("Required keys not implemented")

    def validate_input(self, x: dict) -> None:
        """
        Validates the model input. Supports nested keys using dot notation.
        """
        for key in self.required_input_keys():
            keys = key.split('.')
            current = x
            
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    raise ValueError(f"Missing required input key: {key}")
                current = current[k]

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        raise NotImplementedError("Forward pass not implemented")