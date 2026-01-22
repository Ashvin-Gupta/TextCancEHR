import os
from logging import Logger
import torch
from tqdm import tqdm
from src.training.utils import build_warmup_cosine_scheduler, get_nested_value, create_nested_dict

def train(
        model: torch.nn.Module,
        experiment_dir: str,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimiser: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        device: torch.device,
        epochs: int = 10,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        logger: Logger = None,
    ) -> None:
    """
    Trains a model and saves outputs to experiment_dir.

    Args:
        model (torch.nn.Module): The model to train.
        experiment_dir (str): The directory to save the experiment results.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
        optimiser (torch.optim.Optimiser): The optimiser to use for training.
        loss_function (torch.nn.Module): The loss function to use for training.
        device (torch.device): The device to train on.
        epochs (int): The number of epochs to train for.
        reduce_lr_on_plateau (bool): Whether to reduce the learning rate on plateau.
        logger (Logger): The logger to use for training.

    Returns:
        None
    """

    # create loss tracking
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")

    # create loss log file
    loss_log_filepath = os.path.join(experiment_dir, "loss.log")
    with open(loss_log_filepath, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    model.to(device)

    epoch_pb = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_pb:
        logger.info(f" - Starting epoch {epoch} of {epochs}")

        model.train()

        # train
        train_pb = tqdm(train_dataloader, desc="Training", leave=False)
        for idx, batch in enumerate(train_pb):

            # create input dictionary
            flat_input_dict = {}
            for key in model.required_input_keys():
                value = get_nested_value(batch, key)
                flat_input_dict[key] = value.to(device)
            
            # convert to nested structure that model expects
            input_dict = create_nested_dict(flat_input_dict)

            # create target
            targets = get_nested_value(batch, "ehr.target_token_ids").to(device).view(-1) # (batch_size * sequence_length)

            # forward pass
            optimiser.zero_grad()
            logits = model(input_dict)  # (batch_size, sequence_length, vocab_size)
            logits = logits.view(-1, logits.shape[-1])  # (batch_size * sequence_length, vocab_size)
            loss = loss_function(logits, targets)
            loss.backward()
            optimiser.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            train_loss.append(loss.item())

            # log every 10% of the way through the epoch
            if idx % (len(train_dataloader) // 10) == 0 and logger is not None:
                last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimiser.param_groups[0]['lr']
                logger.info(
                    f"  -- Completed training batch {idx} of {len(train_dataloader)} ({idx / len(train_dataloader) * 100:.2f}%) | mean running train loss: {sum(train_loss) / len(train_loss)} | current lr: {last_lr}"
                )

        # evaluate
        model.eval()
        with torch.no_grad():
            logger.info(" - Starting evaluation")

            val_pb = tqdm(val_dataloader, desc="Evaluating", leave=False)
            for idx, batch in enumerate(val_pb):

                # create input dictionary
                flat_input_dict = {}
                for key in model.required_input_keys():
                    value = get_nested_value(batch, key)
                    flat_input_dict[key] = value.to(device)
                
                # convert to nested structure that model expects
                input_dict = create_nested_dict(flat_input_dict)

                # create target
                targets = get_nested_value(batch, "ehr.target_token_ids").to(device).view(-1) # (batch_size * sequence_length)

                # forward pass
                logits = model(input_dict) # (batch_size, sequence_length, vocab_size)
                logits = logits.view(-1, logits.shape[-1])  # (batch_size * sequence_length, vocab_size)
                loss = loss_function(logits, targets)
                val_loss.append(loss.item())

                # log every 10% of the way through the epoch
                if idx % (len(val_dataloader) // 10) == 0 and logger is not None:
                    logger.info(
                        f"  -- Completed evaluation batch {idx} of {len(val_dataloader)} ({idx / len(val_dataloader) * 100:.2f}%) | mean running val loss: {sum(val_loss) / len(val_loss)}"
                    )

        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_val_loss = sum(val_loss) / len(val_loss)

        if logger is not None:
            logger.info(
                f" - Completed epoch {epoch} of {epochs} | mean train loss: {avg_train_loss} | mean val loss: {avg_val_loss}"
            )

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pth"))
            if logger is not None:
                logger.info(f" - Saved best model with new best val loss: {best_val_loss}")

        # log loss to file
        with open(loss_log_filepath, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        epoch_pb.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
