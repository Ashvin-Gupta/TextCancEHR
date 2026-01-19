#!/bin/bash
#$ -cwd                 
#$ -pe smp 24
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=2
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR/HPC_Classifier/logo/
#$ -e /data/home/qc25022/TextCancEHR/HPC_Classifier/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd) Pretrain with Lora"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# # Run the fine-tuning script
torchrun --nproc_per_node=2 src/cli/finetune_classifier.py --config_filepath configs/llm_classify_with_lora.yaml
# torchrun --nproc_per_node=2 src/pipelines/finetune_llm_classifier.py \
#     --config_filepath configs/llm_classify_with_lora.yaml


echo "Classification fine-tuning complete!"