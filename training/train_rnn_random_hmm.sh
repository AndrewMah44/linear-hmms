#!/bin/bash
#SBATCH --job-name=rand_hmm
#SBATCH --output=logs/rand_hmm_%A_%a.out
#SBATCH --error=logs/rand_hmm_%A_%a.err
#SBATCH --array=3
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org

# Load your environment
module --force purge

source ~/venvs/linear-hmms/bin/activate

export LD_LIBRARY_PATH="$(
    find "$VIRTUAL_ENV/lib" \
        -path '*/site-packages/nvidia/*/lib' \
        -type d \
        -printf '%p:'
)"

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

srun --cpu-bind=cores python - <<'PY'
import jax

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

if jax.default_backend() != "gpu":
    raise RuntimeError("JAX did not initialize the GPU backend")
PY

# Run the script with the array index
srun --cpu-bind=cores python train_rnn_random_hmm.py --i "$SLURM_ARRAY_TASK_ID"
