#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem=32
#SBATCH --job-name=EG121225
#SBATCH --output=EG12122501output.log

# Load environment
source ~/.bashrc               # loads conda
conda activate smfish_env

which python
python --version


# Run your script
python /home/qgs8612/planarian_smFISH/run_server.py --config /home/qgs8612/planarian_smFISH/config.yaml