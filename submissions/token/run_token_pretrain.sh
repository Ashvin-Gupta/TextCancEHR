#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l gpu=1
#$ -l h_rt=24:0:0
#$ -l h_vmem=11G
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_New/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_New/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/CancEHR-Training"

export WANDB_API_KEY="3256683a0a9a004cf52e04107a3071099a53038e"

# --- Environment Setup ---
module load cuda/12.4.0-gcc-12.2.0
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source /data/home/qc25022/CancEHR-Training/ssm/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"

# python -m src.pipelines.token_based.pretrain --config src/pipelines/token_based/configs/cprd_decoder_lstm_test.yaml --experiment_name lstm_test
python -m src.pipelines.token_based.pretrain --config src/pipelines/token_based/configs/mamba_base.yaml --experiment_name mamba_base
# python -m src.pipelines.token_based.models.mamba

echo "Pipeline finished."
deactivate

