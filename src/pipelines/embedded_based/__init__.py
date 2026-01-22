"""
Embedding-based pipeline for pre-embedded event sequences.

This pipeline:
1. Creates embeddings from events using sentence transformers
2. Pretrains decoder models on autoregressive prediction
3. Fine-tunes encoder/decoder models for classification
"""

