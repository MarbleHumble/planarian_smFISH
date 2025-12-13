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

# Ensure conda works in non-interactive shell
source /home/qgs8612/.conda/etc/profile.d/conda.sh
conda activate smfish_env
# -----------------------------
# GPU diagnostics (optional but recommended)
# -----------------------------
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== CUDA visible devices ====="
python - <<EOF
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except ImportError:
    print("Torch not installed or not found!")
EOF

# -----------------------------
# Run smFISH pipeline
# -----------------------------
python /home/qgs8612/planarian_smFISH/run_server.py \
  --config /home/qgs8612/planarian_smFISH/config.yaml