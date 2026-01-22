import os
from logging import Logger
from typing import Literal
from torch.utils.data import DataLoader
from .dataset import NightingaleTrainingDataset

def get_dataloader(
        dataset_dir: str,
        batch_size: int,
        shuffle: bool = True,
        sequence_length: int = 100,
        mode: Literal["train", "eval"] = "train",
        num_workers: int = 4,
        insert_static_demographic_tokens: bool = True,
        clinical_notes_dir: str = None,
        clinical_notes_max_note_count: int = None,
        clinical_notes_max_tokens_per_note: int = None,
        logger: Logger | None = None,
    ) -> DataLoader:
    """
    Create a dataloader for a Nightingale dataset.

    Args:
        dataset_dir (str): Directory containing the pickled tokenized data.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        sequence_length (int): Length of the input and target token sequences.
        mode (Literal["train", "eval"]): Dataset mode; changes how data is loaded and returned. Must be "train" or "eval".
        num_workers (int): Number of workers to use for the dataloader.
        logger (Logger): Logger to use for logging.

    Returns:
        dataloader (DataLoader): A dataloader for the Nightingale dataset.
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"Invalid mode: {mode}. Must be one of 'train', 'eval'.")

    dataset = NightingaleTrainingDataset(dataset_dir, mode, sequence_length, insert_static_demographic_tokens, clinical_notes_dir, clinical_notes_max_note_count, clinical_notes_max_tokens_per_note, logger=logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=("Quick test to create a Nightingale dataloader and inspect a batch."))
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to split directory (e.g., /path/to/train)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(f"Dataset directory not found: {args.dataset_dir}")

    dl = get_dataloader(
        dataset_dir=args.dataset_dir,
        batch_size=8,
        shuffle=True,
        sequence_length=128,
        mode="train",
        num_workers=2,
    )

    batch = next(iter(dl))
    print(batch)
