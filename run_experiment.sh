#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=EG121225GPU
#SBATCH --output=EG_new_GPU_LoG_test_output.log

# -----------------------------
# Environment setup
# -----------------------------
module load python-miniconda3
source activate /home/qgs8612/.conda/envs/smfish_env

# -----------------------------
# Diagnostics (VERY IMPORTANT)
# -----------------------------
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== CUDA visible devices ====="
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

# -----------------------------
# Run pipeline
# -----------------------------
python /home/qgs8612/planarian_smFISH/run_server.py \
  --config /home/qgs8612/planarian_smFISH/config.yaml
