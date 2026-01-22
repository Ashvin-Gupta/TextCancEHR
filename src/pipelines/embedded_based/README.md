# Embedded-Based Pipeline

Train transformer encoder/decoder models on pre-embedded event sequences using sentence transformers.

## Overview

This pipeline creates dense embeddings for each medical event using pretrained sentence transformers (e.g., `all-MiniLM-L6-v2`), then trains lightweight transformer models on these embeddings. This approach:

1. **Separates representation from prediction**: Events are embedded once, reused many times
2. **Faster training**: No need to train embeddings, just transformer layers
3. **Efficient**: Small models (few million parameters) vs. large LLMs (billions)
4. **Flexible**: Supports both encoder (classification) and decoder (generation) models

## Workflow

The pipeline has three stages:

```
1. Create Embeddings  →  2. Pretrain (optional)  →  3. Fine-tune
   (One-time setup)       (Decoder only)             (Classification)
```

### Stage 1: Create Embeddings (One-Time)

Convert tokenized events to dense embeddings:

```bash
# Step 1a: Create vocabulary embeddings (one-time)
python -m src.pipelines.embedded_based.create_vocab_embeddings \
    --vocab_file /path/to/vocab.csv \
    --output_file embeddings/vocab_embeddings.pt \
    --model_name sentence-transformers/all-MiniLM-L6-v2

# Step 1b: Create patient embeddings
python -m src.pipelines.embedded_based.create_embeddings \
    --config_filepath src/pipelines/embedded_based/configs/create_embeddings.yaml
```

This creates:
```
embeddings/
├── train/
│   ├── patient_0.pt
│   ├── patient_1.pt
│   └── ...
├── tuning/
│   └── ...
└── held_out/
    └── ...
```

Each `.pt` file contains:
```python
{
    'embeddings': torch.Tensor,  # (N, 768) - embedded events
    'token_ids': torch.Tensor,   # (N,) - original token IDs
    'label': torch.Tensor        # Classification label
}
```

### Stage 2: Pretrain Decoder (Optional)

Pretrain a decoder model on autoregressive next-event prediction:

```bash
python -m src.pipelines.embedded_based.pretrain \
    --config src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml
```

This is similar to language model pretraining but on event sequences. The pretrained decoder can then be fine-tuned for classification.

### Stage 3: Fine-tune for Classification

Train an encoder or decoder for classification:

```bash
# Encoder (recommended for classification)
python -m src.pipelines.embedded_based.finetune \
    --config src/pipelines/embedded_based/configs/finetune_encoder_embedded.yaml

# Or decoder (if pretrained)
python -m src.pipelines.embedded_based.finetune \
    --config src/pipelines/embedded_based/configs/finetune_decoder_embedded.yaml
```

### Full Pipeline Script

Run all stages at once:

```bash
./run_embedded_pipeline.sh \
    src/pipelines/embedded_based/configs/create_embeddings.yaml \
    src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml \
    src/pipelines/embedded_based/configs/finetune_encoder_embedded.yaml
```

## Configuration

### Stage 1: Create Embeddings

```yaml
model:
  name: sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384  # Depends on model

data:
  data_dir: /path/to/tokenized/data/
  vocab_filepath: /path/to/vocab.csv
  labels_filepath: /path/to/labels.csv
  medical_lookup_filepath: /path/to/medical_lookup.csv
  lab_lookup_filepath: /path/to/lab_lookup.csv
  vocab_embedding_path: embeddings/vocab_embeddings.pt
  embedding_output_dir: embeddings/patients/

device: cuda
```

### Stage 2: Pretrain Decoder

```yaml
model:
  type: transformer_decoder_embedded
  embedding_dim: 768  # From sentence transformer
  model_dim: 256      # Internal dimension
  n_heads: 8
  n_layers: 6
  dropout: 0.1
  max_length: 512
  vocab_size: 10000

data:
  embedding_output_dir: embeddings/patients/

training:
  task: autoregressive
  output_dir: results/embedded_based/pretrain/
  batch_size: 32
  eval_batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  min_learning_rate: 0.000001
  num_workers: 4
  save_every: 10
```

### Stage 3: Fine-tune Encoder

```yaml
model:
  type: transformer_encoder_embedded
  embedding_dim: 768
  model_dim: 256
  n_heads: 8
  n_layers: 6
  dropout: 0.1
  max_length: 512
  num_classes: 2

data:
  embedding_output_dir: embeddings/patients/

training:
  output_dir: results/embedded_based/finetune_encoder/
  pretrained_checkpoint: null  # Or path to pretrained model
  batch_size: 32
  eval_batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  early_stopping_patience: 10
  scheduler_patience: 3
  class_weights: [0.5, 1.5]  # Optional: for imbalanced data
  num_workers: 4
```

## Models

### Transformer Encoder Embedded

- **Architecture**: Bidirectional transformer encoder
- **Use case**: Classification tasks
- **Input**: Pre-embedded event sequence
- **Output**: Classification logits

```
Input Embeddings (N, 768) 
  → Linear Projection (768 → model_dim)
  → Positional Encoding
  → N x Transformer Encoder Blocks
  → Mean Pooling
  → Classification Head
  → Logits (num_classes)
```

### Transformer Decoder Embedded

- **Architecture**: Causal transformer decoder
- **Use cases**: 
  - Pretraining: Next-event prediction
  - Fine-tuning: Classification (with pooling head)
- **Input**: Pre-embedded event sequence
- **Output**: Next-event logits or classification logits

```
Input Embeddings (N, 768)
  → Linear Projection (768 → model_dim)
  → Positional Encoding
  → N x Transformer Decoder Blocks (causal mask)
  → Linear Head (→ vocab_size or num_classes)
```

## Embedding Models

Popular sentence transformer models:

| Model | Embedding Dim | Speed | Quality | Use Case |
|-------|--------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Quick experiments |
| `all-mpnet-base-v2` | 768 | Medium | Better | Balanced |
| `all-MiniLM-L12-v2` | 384 | Medium | Better | Balanced |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Slow | Best | Maximum quality |

Change in `create_embeddings.yaml`:

```yaml
model:
  name: sentence-transformers/all-mpnet-base-v2
  embedding_dim: 768  # Update to match model
```

## Output

### Embeddings

```
embeddings/patients/
├── train/
│   ├── patient_0.pt  # {'embeddings': (N,768), 'token_ids': (N,), 'label': scalar}
│   ├── patient_1.pt
│   └── ...
├── tuning/
└── held_out/
```

### Pretrained Model

```
results/embedded_based/pretrain/
├── config.yaml
├── best_checkpoint.pt
├── latest_checkpoint.pt
└── checkpoint_epoch_10.pt
```

### Fine-tuned Model

```
results/embedded_based/finetune_encoder/
├── config.yaml
├── best_checkpoint.pt  # Best F1 score
└── latest_checkpoint.pt
```

## Loading Models

```python
import torch
from src.pipelines.embedded_based.models.transformer_encoder_embedded import TransformerEncoderEmbedded

# Load config
config = {
    'type': 'transformer_encoder_embedded',
    'embedding_dim': 768,
    'model_dim': 256,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'max_length': 512,
    'num_classes': 2
}

# Create model
model = TransformerEncoderEmbedded(config)

# Load checkpoint
checkpoint = torch.load('results/embedded_based/finetune_encoder/best_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
embeddings = ...  # Your embedded sequence (N, 768)
padding_mask = ...  # Mask for padding (N,)

with torch.no_grad():
    logits = model({
        'embeddings': embeddings.unsqueeze(0),  # (1, N, 768)
        'padding_mask': padding_mask.unsqueeze(0)  # (1, N)
    })
    prediction = logits.argmax(-1)
```

## Benefits

1. **Fast iteration**: Create embeddings once, train many models
2. **Efficient**: Small models (10-50M params) vs. LLMs (7B+ params)
3. **Flexible**: Easy to try different transformer architectures
4. **Pretraining**: Optional decoder pretraining for better initialization
5. **Quality**: Sentence transformers capture semantic meaning

## Limitations

1. **Fixed embeddings**: Can't fine-tune embedding model
2. **Two-stage**: Requires embedding creation before training
3. **Memory**: Stores all embeddings on disk
4. **Less expressive**: Fixed 768-dim embeddings vs. learned representations

## Tips

### Memory Optimization

- Use smaller sentence transformer (e.g., `all-MiniLM-L6-v2` instead of mpnet)
- Reduce `model_dim` in transformer config
- Reduce `batch_size`

### Quality Improvement

- Use better sentence transformer (e.g., `all-mpnet-base-v2`)
- Increase `model_dim` and `n_layers`
- Pretrain decoder before fine-tuning
- Use class weights for imbalanced data

### Speed Optimization

- Create embeddings on GPU (`device: cuda`)
- Increase `num_workers` for dataloaders
- Use smaller models during experimentation
- Cache vocabulary embeddings

## Comparison with Other Pipelines

| Feature | Token-Based | Text-Based | Embedded-Based |
|---------|-------------|------------|----------------|
| **Training Speed** | Medium | Slow | Fast |
| **Model Size** | Medium (100M) | Large (7B+) | Small (10-50M) |
| **Iteration Speed** | Medium | Slow | Fast |
| **Quality** | Good | Best | Good |
| **Flexibility** | High | Medium | High |
| **Memory** | Medium | High | Low (training) / High (storage) |

## Troubleshooting

**Issue**: Embeddings creation is slow
- Use GPU: `device: cuda`
- Reduce batch size if OOM
- Use smaller sentence transformer model
- Pre-compute vocabulary embeddings once

**Issue**: Model not improving
- Try pretraining decoder first
- Increase `model_dim` or `n_layers`
- Check for class imbalance (use `class_weights`)
- Verify embeddings are created correctly

**Issue**: Out of memory during training
- Reduce `batch_size`
- Reduce `model_dim`
- Reduce `n_layers`
- Use smaller sequence length

**Issue**: Embeddings take too much disk space
- Use smaller sentence transformer (384-dim instead of 768-dim)
- Store embeddings as float16 instead of float32
- Compress with sparse representations

## Resources

- [Sentence Transformers](https://www.sbert.net/)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [All-MiniLM Paper](https://arxiv.org/abs/2002.10957)

