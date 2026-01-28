#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=0:10:0
#$ -l h_vmem=11G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR/HPC_Interpretability/logo/
#$ -e /data/home/qc25022/TextCancEHR/HPC_Interpretability/loge/

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

echo "Starting experiment from directory: $(pwd) Interpretability"

# # Run the fine-tuning script
python -m src.pipelines.text_based.analyze_classifier_interpretability \
--config_filepath src/pipelines/text_based/configs/llm_classify_pretrained_cls_lora.yaml \
--checkpoint_path /data/scratch/qc25022/pancreas/experiments/lora-6-month-logistic-raw/checkpoint-7856 \
--output_dir ./interpretability_results

echo "Interpretability analysis complete!"