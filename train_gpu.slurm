#!/bin/sh
#SBATCH --account=ie-idi             # E.g. ie-idi
#SBATCH --job-name=odelia_train
#SBATCH --time=0-08:00:00               # format: D-HH:MM:SS

#SBATCH --partition=GPUQ                # Requesting a GPU queue
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=32G                       # Total RAM
#SBATCH --nodes=1
#SBATCH --output=slurm-%j.out           # stdout
#SBATCH --error=slurm-%j.err            # stderr

#SBATCH --mail-user=vebjorel@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}" || exit 1

echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "ID of job: ${SLURM_JOB_ID}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda/2024.02-1

# Optional: uncomment and set if your cluster requires activating a specific conda env.
# source activate <conda_env_name>

# Train the model using project defaults/data layout.
python -u cli.py train \
  --data-root ODELIA2025/data \
  --fit-splits train val test \
  --artifact model_cnn.pt \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --monitor-every 20 \
  --monitor-csv-path train_monitor.csv


