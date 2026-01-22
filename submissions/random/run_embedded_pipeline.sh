#!/bin/bash
#$ -cwd                 
#$ -pe smp 4
#$ -l h_rt=1:0:0
#$ -l h_vmem=12G
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_New/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_New/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/CancEHR-Training"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting run embedded pipeline experiment from directory: $(pwd)"

# Create vocabulary embeddings
# python -m src.pipelines.embedded_based.create_vocab_embeddings --config_filepath src/pipelines/embedded_based/configs/create_embeddings.yaml
# Create embedding corpus
# python -m src.pipelines.embedded_based.create_embeddings --config_filepath src/pipelines/embedded_based/configs/create_embeddings.yaml
# Pretrain encoder
python -m src.pipelines.embedded_based.pretrain --config_filepath src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml
# python -m src.pipelines.embedded_based.models.transformer_decoder_embedded

echo "Pipeline finished."
deactivate

