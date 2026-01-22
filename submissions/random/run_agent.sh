#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=240:0:0       # 2 hours per run (1 epoch), 1 for debugging
#$ -l h_vmem=11G
#$ -l gpu=1
#$ -o /data/home/qc25022/CancEHR-Training/HPC_Files/logo/
#$ -e /data/home/qc25022/CancEHR-Training/HPC_Files/loge/
#$ -j n

cd /data/home/qc25022/CancEHR-Training/
export PYTHONPATH="/data/home/qc25022/CancEHR-Training/:$PYTHONPATH"

# 1. Activate your environment
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# 2. Define your Sweep ID (get this from the 'wandb sweep' command)
SWEEP_ID="ashvingupta00-imperial-college-london/CancEHR-Training-src_experiments/oj00twwx"

# 4. Run the agent
# It will execute this command over and over with new sweep params
wandb agent --count 10 ${SWEEP_ID}
