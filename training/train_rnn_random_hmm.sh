#!/bin/bash
#SBATCH --job-name=rand_hmm
#SBATCH --output=logs/rand_hmm_%A_%a.out
#SBATCH --error=logs/rand_hmm_%A_%a.err
#SBATCH --array=2
#SBATCH --cpus-per-task=1         # Adjust CPU count as needed
#SBATCH --mem=14G                 # Adjust memory as needed
#SBATCH --time=5:00:00            # Max runtime
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=amah@flatironinstitute.org  # Or your actual email
#SBATCH --partition=genx

# Load your environment
module --force purge
module load python
source ~/venvs/linear-hmms/bin/activate

# Run the script with the array index
python3 train_rnn_random_hmm.py --i $SLURM_ARRAY_TASK_ID
