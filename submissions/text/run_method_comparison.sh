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
LR_DIR="./interpretability_results"
ABLATION_DIR="./ablation_results"
IG_DIR="./ig_results"
OUTPUT_DIR="./method_comparison"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo ""
echo "Configuration:"
echo "  LR Results:       $LR_DIR"
echo "  Ablation Results: $ABLATION_DIR"
echo "  IG Results:       $IG_DIR"
echo "  Output:           $OUTPUT_DIR"
echo ""

# Run comparison
echo "Running method comparison..."
python -m src.pipelines.text_based.compare_interpretability_methods \
    --lr_dir $LR_DIR \
    --ablation_dir $ABLATION_DIR \
    --ig_dir $IG_DIR \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Comparison completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh $OUTPUT_DIR
else
    echo ""
    echo "✗ Comparison failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "End time: $(date)"
echo "=========================================="

