#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -l h_vmem=11G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR/HPC_Motor/logo/
#$ -e /data/home/qc25022/TextCancEHR/HPC_Motor/loge/

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

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Run MOTOR fine-tuning
echo "Starting MOTOR Time-to-Event Fine-tuning..."
echo "Config: src/pipelines/text_based/configs/motor_finetune.yaml"
echo "========================================"

python -m src.pipelines.text_based.finetune_motor \
    --config_filepath src/pipelines/text_based/configs/motor_finetune.yaml

echo "========================================"
echo "MOTOR training complete!"
