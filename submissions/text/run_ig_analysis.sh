#!/bin/bash
#SBATCH --job-name=ig_analysis
#SBATCH --output=logs/ig_analysis_%j.out
#SBATCH --error=logs/ig_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Integrated Gradients Analysis Script
# Computes IG attributions for LLM classifier

echo "=========================================="
echo "Integrated Gradients Analysis"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

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

