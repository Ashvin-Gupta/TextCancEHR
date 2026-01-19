#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
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

echo "Starting experiment from directory: $(pwd) No Pretrain"

# # Run the fine-tuning script
python -m src.cli.finetune_classifier --config_filepath configs/llm_classify_no_pretrain.yaml
# python -m src.pipelines.finetune_llm_classifier \
# --config_filepath configs/llm_classify_no_pretrain.yaml #no pretrain, only classifier

echo "Classification fine-tuning complete!"
