#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem=32
#SBATCH --job-name=EG121225GPU
#SBATCH --output=EG_new_GPU_LoG_test_output.log

# Load environment
$ module load python-miniconda3
$ source activate /home/qgs8612/.conda/envs/smfish_env


# Run your script
python /home/qgs8612/planarian_smFISH/run_server.py --config /home/qgs8612/planarian_smFISH/config.yaml