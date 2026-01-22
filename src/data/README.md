# Data module

Nightingale expects tokenized EHR trajectories produced by the `ehr-tokenization` pipeline. This module provides dataset and dataloader utilities that enable training and evaluation.

## Directory format (input)
Each split directory (e.g., `train/`, `tuning/`, and `held_out/`) contains one or more `.pkl` files. Each pickle file is a list of subject dicts:

- `subject_id` (int)
- `tokens` (list[int])  — token IDs
- `timestamps` (list[int|float]) — aligned timestamps (same length as tokens)

Example structure:
```
<dataset_root>/
  train/
    part-000.pkl
    part-001.pkl
  tuning/
    part-000.pkl
  vocab.csv           # columns: token, str, count
```

## Datasets

### `NightingaleTrainingDataset`
- Purpose: Creates length-`sequence_length` windows for autoregressive next-token training. Windows use random starting_indexes during training.
- Modes:
  - `train`: stores full subject sequences and samples a random start index per item; returns overlapping input/target pairs.
  - `eval`: pre-chunks each subject into deterministic non-overlapping windows (length `sequence_length`).
- Filtering: subjects shorter than `sequence_length + 1` are filtered out of the datasets.

Item format:
- `subject_id`: `torch.Tensor` scalar
- `input_tokens`: `(S,)` tokens `[t0..tS-1]`
- `target_tokens`: `(S,)` tokens `[t1..tS]`
- `input_timestamps`, `target_timestamps`: `(S,)`

Where `S` is the length of the context window and `tx` is the `x`th index of the sequence.

Constructor:
```python
NightingaleTrainingDataset(
    data_dir: str,
    mode: str,                 # "train" | "eval"
    sequence_length: int = 100,
    logger: Optional[Logger] = None,
)
```

### `NightingaleEvaluationDataset`
- Purpose: simple subject-level access plus token/string conversion utilities for analysis and simulation.
- Currently hardcoded to just load the first 2 `.pkl` files for now for faster evaluation (this will be changed in the future for bigger evaluations).
- Provides:
  - `get_data_by_subject_id(subject_id: int) -> dict`
  - `token_to_string(token: int) -> str`
  - `string_to_token(text: str) -> int`
  - `tokens_to_strings(tokens: torch.Tensor) -> list[str]`

Constructor:
```python
NightingaleEvaluationDataset(
    data_dir: str,
    vocab_path: str,
    logger: Optional[Logger] = None,
)
```

## Dataloader
Convenience factory for training/eval loaders:
```python
from src.data.dataloader import get_dataloader

train_loader = get_dataloader(
    dataset_dir="/path/to/train",
    batch_size=64,
    shuffle=True,
    sequence_length=512,
    mode="train",
    num_workers=4,
)

val_loader = get_dataloader(
    dataset_dir="/path/to/tuning",
    batch_size=64,
    shuffle=True,
    sequence_length=512,
    mode="eval",
    num_workers=4,
)
```
Returns batches with:
```python
batch["subject_id"]           # (B,)
batch["input_tokens"]         # (B, S)
batch["target_tokens"]        # (B, S)
batch["input_timestamps"]     # (B, S)
batch["target_timestamps"]    # (B, S)
```

## Training objective (for context)
Given `input_tokens` of length `S`, the model is trained to predict each next token in `target_tokens` via cross-entropy over all positions.