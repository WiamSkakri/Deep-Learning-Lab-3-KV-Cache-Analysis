#!/bin/bash
#SBATCH --job-name=kv_cache_lab3
#SBATCH --output=logs/kv_cache_%j.out
#SBATCH --error=logs/kv_cache_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

echo "=============================================="
echo "Lab 3: KV Cache Analysis"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=============================================="

# Activate virtual environment
cd ~/kv_lab
source venv/bin/activate

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run the script
echo "Running script.py..."
echo "=============================================="
python script.py

echo ""
echo "=============================================="
echo "Job finished at: $(date)"
echo "=============================================="
echo ""
echo "Output files:"
ls -la lab3_report.txt 2>/dev/null || echo "Report file not found"
