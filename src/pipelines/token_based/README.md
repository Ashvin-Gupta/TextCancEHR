# Token-Based Pipeline

Train custom transformer decoder, LSTM, and GPT-2 models directly on integer token sequences.

## Overview

This pipeline trains models on tokenized EHR sequences where each medical event (diagnosis, prescription, lab test, etc.) is represented as an integer token. Models learn autoregressive next-token prediction and can be fine-tuned for classification tasks.

## Models

### Available Architectures

1. **LSTM** (`lstm`)
   - Bidirectional LSTM with embedding layer
   - Good baseline for sequence modeling
   - Fast training, lower memory footprint

2. **Transformer Decoder** (`transformer`)
   - GPT-style decoder-only transformer
   - Causal self-attention
   - Configurable layers, heads, and dimensions

3. **GPT-2** (`gpt2`)
   - Full GPT-2 implementation (adapted from nanoGPT)
   - Supports various model sizes
   - Flash attention support (PyTorch >= 2.0)

4. **LSTM with Clinical Notes** (`lstm_note`)
   - LSTM that processes both structured events and clinical text
   - Separate encoding pathway for notes

All models extend `BaseNightingaleModel` from `src/pipelines/shared/base_models.py`.

## Usage

### Training Script

```bash
python -m src.pipelines.token_based.pretrain \
    --config path/to/config.yaml \
    --experiment_name my_experiment
```

Or use the convenience script:

```bash
./run_token_pretrain.sh path/to/config.yaml my_experiment
```

### Configuration

Example config (`configs/encoder_lstm.yaml`):

```yaml
name: lstm_baseline

model:
  type: lstm
  vocab_size: 10000
  embedding_dim: 256
  hidden_dim: 512
  n_layers: 2
  dropout: 0.1

data:
  train_dataset_dir: /path/to/train/
  val_dataset_dir: /path/to/val/
  vocab_path: /path/to/vocab.csv
  batch_size: 32
  sequence_length: 512
  shuffle: true
  insert_static_demographic_tokens: true

optimiser:
  type: adamw
  lr: 0.0001
  scheduler:
    type: warmup_cosine
    warmup_steps: 1000
    lr_min_ratio: 0.1

loss_function:
  type: cross_entropy

training:
  epochs: 50
  device: cuda
```

### Key Configuration Options

**Model:**
- `type`: Model architecture (`lstm`, `transformer`, `gpt2`, `lstm_note`)
- `vocab_size`: Total number of unique tokens in vocabulary
- Architecture-specific params (layers, dimensions, heads, etc.)

**Data:**
- `train_dataset_dir`: Directory with training `.pkl` files
- `val_dataset_dir`: Directory with validation `.pkl` files
- `vocab_path`: Path to vocabulary CSV
- `sequence_length`: Maximum sequence length (tokens)
- `batch_size`: Training batch size
- `insert_static_demographic_tokens`: Include age, gender, ethnicity tokens

**Optimizer:**
- `type`: `adam` or `adamw`
- `lr`: Learning rate
- `scheduler`: Optional learning rate scheduling

**Training:**
- `epochs`: Number of training epochs
- `device`: `cuda` or `cpu`

### Clinical Notes Support

To train with clinical notes (LSTM Note model):

```yaml
model:
  type: lstm_note
  # ... other params

data:
  # ... other params
  clinical_notes:
    dir: /path/to/notes/
    max_note_count: 10
    max_tokens_per_note: 512
```

## Output

Results are saved to `results/token_based/{experiment_name}/`:

```
results/token_based/my_experiment/
├── config.yaml           # Experiment configuration
├── training.log          # Training logs
├── loss.log             # Loss values per epoch
├── model.pth            # Best model checkpoint
├── vocab.csv            # Vocabulary (copied from config)
└── checkpoints/         # Periodic checkpoints
```

## Data Format

Expected input format (from CancEHR-tokenization):

**Training data** (`.pkl` files):
```python
[
    {
        'subject_id': int,
        'tokens': List[int],      # Token IDs
        'timestamps': List[float], # Unix timestamps
        'label': int              # Classification label (optional)
    },
    ...
]
```

**Vocabulary** (`.csv`):
```csv
token,str
0,<start>
1,<end>
2,AGE_decile_5
3,MEDICAL//C10  # Diagnosis code
4,LAB//GLUCOSE  # Lab test
...
```

## Model Loading

To load a trained model:

```python
import torch
from src.pipelines.token_based.models.lstm import LSTM

# Load config
config = {...}  # Your model config

# Create model
model = LSTM(config)

# Load weights
checkpoint = torch.load('results/token_based/my_experiment/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Extending

### Adding a New Model

1. Create model file in `models/`:

```python
# models/my_model.py
from src.pipelines.shared.base_models import BaseNightingaleModel

class MyModel(BaseNightingaleModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        # ... your architecture
    
    def required_config_keys(self) -> set[str]:
        return {"vocab_size", "embedding_dim", ...}
    
    def required_input_keys(self) -> set[str]:
        return {"ehr.input_token_ids"}
    
    def forward(self, x: dict) -> torch.Tensor:
        # ... forward pass
        return logits
```

2. Update `pretrain.py` to import your model:

```python
from src.pipelines.token_based.models.my_model import MyModel

def load_model(model_config: dict):
    # ...
    elif model_type == "my_model":
        model = MyModel(model_config)
    # ...
```

3. Create config file in `configs/my_model.yaml`

4. Test:

```bash
./run_token_pretrain.sh src/pipelines/token_based/configs/my_model.yaml test_exp
```

## Tips

- **Memory**: Reduce `batch_size` or `sequence_length` if OOM
- **Speed**: Use `device: cuda` and enable flash attention (GPT-2)
- **Overfitting**: Increase `dropout`, reduce model size, or use regularization
- **Baseline**: Start with LSTM before trying complex architectures

## Troubleshooting

**Issue**: `ValueError: Model type X not supported`
- Check `model.type` in config matches available models
- Ensure model is imported in `pretrain.py`

**Issue**: `Missing required field in data`
- Verify all required config fields are present
- Check paths point to existing data directories

**Issue**: CUDA out of memory
- Reduce `batch_size`
- Reduce `sequence_length`
- Use smaller model (`n_layers`, `hidden_dim`)
- Enable gradient checkpointing (GPT-2)

## References

- Base transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Implementation adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)

