#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=EG121225GPUfastlog
#SBATCH --output=EG_fastLoG_GPU_LoG_test_output.log

module load python-miniconda3

# Option A: source bashrc + activate (works if bashrc is set up)
# source ~/.bashrc
# conda activate /home/qgs8612/.conda/envs/smfish_env

# Option B: direct Python from environment (recommended)
PYTHON=/home/qgs8612/.conda/envs/smfish_env/bin/python

# GPU diagnostics
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== CUDA visible devices ====="
$PYTHON - <<EOF
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF


# Run smFISH pipeline
$PYTHON /home/qgs8612/planarian_smFISH/run_server.py \
    --config /home/qgs8612/planarian_smFISH/config.yaml