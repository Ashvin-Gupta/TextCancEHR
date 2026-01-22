#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/CancEHR-Training/HPC_Pretrain/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_Pretrain/loge/

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

echo "Starting experiment from directory: $(pwd) No Pretrain"

# # Run the fine-tuning script
python -m src.pipelines.text_based.finetune_llm_classifier \
--config_filepath src/pipelines/text_based/configs/llm_classify_no_pretrain.yaml #no pretrain, only classifier
# --config_filepath src/pipelines/text_based/configs/llm_classify_pretrained_cls_lora.yaml #pretrained, classifier + lora
   # --config_filepath src/pipelines/text_based/configs/llm_classify_pretrained_cls.yaml #pretrained, only classifier
   
   

# Run the evaluation script
# python -m src.pipelines.text_based.evaluate_checkpoint_classifier \
#    --config_filepath src/pipelines/text_based/configs/llm_finetune_classifier.yaml

echo "Classification fine-tuning complete!"


