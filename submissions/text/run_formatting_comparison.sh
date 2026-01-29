#!/bin/bash
#$ -cwd                 
#$ -pe smp 4
#$ -l h_rt=1:0:0
#$ -l h_vmem=8G
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR/HPC_Interpretability/logo/
#$ -e /data/home/qc25022/TextCancEHR/HPC_Interpretability/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
cd "${BASE_DIR}"

echo "Starting Case vs Control Formatting Comparison..."
echo "================================== ======"

# Run the formatting comparison script
# You can add --max_samples 1000 to only analyze first 1000 of each group (for testing)
python -m src.pipelines.text_based.compare_case_control_formatting \
    --config_filepath src/pipelines/text_based/configs/llm_classify_pretrained_cls_lora.yaml

echo "========================================"
echo "Comparison complete! Check case_control_formatting_comparison.csv"
