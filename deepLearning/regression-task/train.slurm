#!/bin/bash
#SBATCH -A naiss2026-4-1221-gpu      # <-- replace with your project account
#SBATCH -p gpu        
#SBATCH -J train 
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -o hello_%j.out
#SBATCH -e hello_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "User: $USER"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

module load GPU/Python/3.13.5-bare-gcc-2025b-eb
module load GPU/buildenv-gcccuda/2026.03-cu13.0

# nvidia-smi
#rm hello*out hello*err
source /home/samuelf/github/viralDNAmodelisationProjectDeepLearning/deepLearning/regression-task/torch_env/bin/activate

echo "Python: $(which python)"
#nvidia-smi
python - <<'EOF'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
EOF

# Force more frequent writes to stdout:
export PYTHONUNBUFFERED=1
python /home/samuelf/github/viralDNAmodelisationProjectDeepLearning/deepLearning/regression-task/trainingWithMeta.py -nd /nobackup/proj/disk/naiss2026-4-1221/shared/robin-cohen-data-alvis/noisy/ -nnd /nobackup/proj/disk/naiss2026-4-1221/shared/robin-cohen-data-alvis//noNoise/ -m /nobackup/proj/disk/naiss2026-4-1221/shared/robin-cohen-data-alvis/join.csv  -mo /home/samuelf/github/viralDNAmodelisationProjectDeepLearning/deepLearning/regression-task/out/best_model.pth

# sleep 10

echo "Done."
