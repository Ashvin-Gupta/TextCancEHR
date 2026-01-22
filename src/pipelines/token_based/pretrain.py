"""
Token-based pretraining/fine-tuning script.

Trains custom transformer decoder, LSTM, or GPT-2 models, or mamba model on integer token sequences.
This script handles both pretraining and fine-tuning depending on the task configuration.

Usage:
    python -m src.pipelines.token_based.pretrain --config path/to/config.yaml --experiment_name my_experiment
"""

import logging
import os
import shutil
import pandas as pd
import torch
import yaml
from datetime import datetime
import wandb

from src.data.dataloader import get_dataloader
from src.training.token_trainer import train
from src.training.utils import build_warmup_cosine_scheduler

# Import token-based models
from src.pipelines.token_based.models.transformer_decoder import TransformerDecoder
from src.pipelines.token_based.models.lstm import LSTM
from src.pipelines.token_based.models.gpt2 import GPT2
from src.pipelines.token_based.models.mamba import MambaDecoder

def create_logger(experiment_dir: str, experiment_name: str) -> logging.Logger:
    """
    Creates a logger for an experiment.

    Args:
        experiment_dir (str): The directory to save the log file.
        experiment_name (str): The name of the experiment.

    Returns:
        logger (logging.Logger): The logger for the experiment.
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # create file handler
    file_handler = logging.FileHandler(os.path.join(experiment_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    return logger


def load_model(model_config: dict) -> torch.nn.Module:
    """
    Loads a token-based model from configuration.

    Args:
        model_config (dict): The model configuration.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model_type = model_config["type"]
    
    if model_type == "lstm":
        model = LSTM(model_config)
    elif model_type == "transformer":
        model = TransformerDecoder(model_config)
    elif model_type == "gpt2":
        model = GPT2(model_config)
    elif model_type == "mamba":
        model = MambaDecoder(model_config)
    else:
        raise ValueError(f"Model type {model_type} not supported in token-based pipeline")

    return model


def run_experiment(config_path: str, experiment_name: str) -> None:
    """
    Runs a token-based training experiment with the given config and experiment name.

    Args:
        config_path (str): The path to the config file.
        experiment_name (str): The name of the experiment.

    Experiment results are saved to results/token_based/{experiment_name}.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate config before running experiment
    validate_config(config)

    # Define experiment directory
    experiment_dir = os.path.join("results/token_based", experiment_name)

    # check if experiments directory exists, if so ask user if they want to overwrite
    if os.path.exists(experiment_dir):
        overwrite = "y"
        if overwrite != "y":
            print("Exiting...")
            return
        else:
            shutil.rmtree(experiment_dir)

    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # create experiment logger
    logger = create_logger(experiment_dir, experiment_name)

    # Set up WandB (add this after config loading, before training)
    wandb_config = config.get('wandb', {})
    
    # Build default run name from hyperparameters
    model_name = config['model']['type']
    default_run_name = (
        f"{model_name}-pretrain"
        f"_lr{config['optimiser']['lr']}"
        f"_bs{config['data']['batch_size']}"
        f"_epochs{config['training']['epochs']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "token-pretraining")
        run_name = wandb_config.get("run_name", default_run_name)
        
        print(f"wandb module location: {wandb.__file__}")
        print(f"wandb has 'init': {hasattr(wandb, 'init')}")
        
        wandb.init(
            project=wandb_config.get("project", "token-pretraining"),
            config=config,  # Pass entire config to wandb
            name=run_name
        )
        logger.info(f"WandB enabled - Project: {wandb_config.get('project', 'token-pretraining')}, Run: {run_name}")
    

    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Config: {config}")
    logger.info(f"Training on device: {config['training']['device']}")

    # Save config to experiment directory
    config["experiment_metadata"] = {
        "start_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # save the vocab to the experiment directory
    vocab_df = pd.read_csv(config["data"]["vocab_path"])
    vocab_df.to_csv(os.path.join(experiment_dir, "vocab.csv"), index=False)

    # Create dataloaders
    train_dataloader = get_dataloader(
        config["data"]["train_dataset_dir"],
        config["data"]["batch_size"],
        config["data"]["shuffle"],
        config["data"]["sequence_length"],
        mode="train",
        insert_static_demographic_tokens=config["data"]["insert_static_demographic_tokens"],
        clinical_notes_dir=config["data"]["clinical_notes"]["dir"] if "clinical_notes" in config["data"] else None,
        clinical_notes_max_note_count=config["data"]["clinical_notes"]["max_note_count"] if "clinical_notes" in config["data"] else None,
        clinical_notes_max_tokens_per_note=config["data"]["clinical_notes"]["max_tokens_per_note"] if "clinical_notes" in config["data"] else None,
        logger=logger,
    )
    val_dataloader = get_dataloader(
        config["data"]["val_dataset_dir"],
        config["data"]["batch_size"],
        config["data"]["shuffle"],
        config["data"]["sequence_length"],
        mode="eval",
        insert_static_demographic_tokens=config["data"]["insert_static_demographic_tokens"],
        clinical_notes_dir=config["data"]["clinical_notes"]["dir"] if "clinical_notes" in config["data"] else None,
        clinical_notes_max_note_count=config["data"]["clinical_notes"]["max_note_count"] if "clinical_notes" in config["data"] else None,
        clinical_notes_max_tokens_per_note=config["data"]["clinical_notes"]["max_tokens_per_note"] if "clinical_notes" in config["data"] else None,
        logger=logger,
    )

    # Load model
    model = load_model(config["model"])

    # Create loss function
    if config["loss_function"]["type"] == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function type {config['loss_function']['type']} not supported")

    # Create optimiser
    if config["optimiser"]["type"] == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=config["optimiser"]["lr"],
        )
    elif config["optimiser"]["type"] == "adamw":
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimiser"]["lr"],
        )
    else:
        raise ValueError(f"Optimiser type {config['optimiser']['type']} not supported")

    # Create learning rate scheduler if its specified
    if "scheduler" in config["optimiser"]:
        if config["optimiser"]["scheduler"]["type"] == "warmup_cosine":
            lr_scheduler = build_warmup_cosine_scheduler(
                optimiser,
                config["training"]["epochs"] * len(train_dataloader),
                warmup_steps=config["optimiser"]["scheduler"]["warmup_steps"],
                lr_min_ratio=config["optimiser"]["scheduler"]["lr_min_ratio"],
            )
        else:
            raise ValueError(f"Scheduler type {config['optimiser']['scheduler']['type']} not supported")
    else:
        lr_scheduler = None

    # Run training
    train(
        model=model,
        experiment_dir=experiment_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimiser=optimiser,
        loss_function=loss_function,
        device=config["training"]["device"],
        epochs=config["training"]["epochs"],
        lr_scheduler=lr_scheduler,
        logger=logger,
        wandb_enabled=wandb_config.get("enabled", False),
    )
    if wandb_config.get("enabled", False):
        wandb.finish()


def validate_config(config: dict) -> None:
    """
    Validate that all required fields exist in the config.

    Args:
        config (dict): The config to validate.
    """
    required_fields = {
        "name": str,
        "model": dict,
        "optimiser": dict,
        "loss_function": dict,
        "training": dict,
        "data": dict,
    }

    # Check top-level required fields
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: '{field}'")
        if not isinstance(config[field], expected_type):
            raise ValueError(f"Field '{field}' must be of type {expected_type.__name__}")

    # Validate optimiser section
    optimiser_required = ["type", "lr"]
    for field in optimiser_required:
        if field not in config["optimiser"]:
            raise ValueError(f"Missing required field in optimiser: '{field}'")

    # Validate loss_function section
    if "type" not in config["loss_function"]:
        raise ValueError("Missing required field in loss_function: 'type'")

    # Validate training section
    training_required = ["epochs", "device"]
    for field in training_required:
        if field not in config["training"]:
            raise ValueError(f"Missing required field in training: '{field}'")

    # Validate data section
    data_required = [
        "train_dataset_dir",
        "val_dataset_dir",
        "sequence_length",
        "batch_size",
        "shuffle",
        "insert_static_demographic_tokens",
    ]
    for field in data_required:
        if field not in config["data"]:
            raise ValueError(f"Missing required field in data: '{field}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run token-based training experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    args = parser.parse_args()

    run_experiment(args.config, args.experiment_name)

