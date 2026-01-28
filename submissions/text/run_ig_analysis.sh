#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0
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
# Configuration
CONFIG_PATH="src/pipelines/text_based/configs/llm_classify_pretrained_cls_lora.yaml"
CHECKPOINT_PATH="/data/scratch/qc25022/pancreas/experiments/lora-6-month-logistic-raw/checkpoint-7856"
OUTPUT_DIR="./ig_results"
NUM_SAMPLES=10
N_STEPS=50
BASELINE="pad"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Activate environment (if needed)
# source /path/to/venv/bin/activate

echo ""
echo "Configuration:"
echo "  Config:       $CONFIG_PATH"
echo "  Checkpoint:   $CHECKPOINT_PATH"
echo "  Output:       $OUTPUT_DIR"
echo "  Samples:      $NUM_SAMPLES"
echo "  IG Steps:     $N_STEPS"
echo "  Baseline:     $BASELINE"
echo ""

# Run IG analysis
echo "Running Integrated Gradients analysis..."
python -m src.pipelines.text_based.analyze_integrated_gradients \
    --config_filepath $CONFIG_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --n_steps $N_STEPS \
    --baseline_strategy $BASELINE

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Analysis completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh $OUTPUT_DIR
else
    echo ""
    echo "✗ Analysis failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "End time: $(date)"
echo "=========================================="

