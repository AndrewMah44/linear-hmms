import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time 
import argparse
import datetime
import pickle as pkl
import jax.random as jr

from linear_hmms.models.lru import StackedLinearRNN
from linear_hmms.utils.load_config import load_paths_config
from linear_hmms.utils.generate_hmm_dataset import generate_hmm_dataset
from linear_hmms.training.train_prediction_model import train_prediction_model

# ==== Training Parameters ====
dataset_params = {
    'n_train': 2**20,
    'n_validation': 20_000,
    'n_test': 50_000,
    'num_timesteps': 256,
}

opt_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 64,
    'weight_decay': 1e-4
}

# ==== Task Parameters ====
alpha_ts = [0.01, 0.25, 0.5, 0.75, 1]
alpha_es = [0.01, 0.25, 0.5, 0.75, 1]
class_state_ratio = [0.5, 1, 2]
num_states_vec = [16, 32, 64]

conditions = [(alpha_e, alpha_t, ratio, num_states) 
                for num_states in num_states_vec
                for alpha_e in alpha_es
                for alpha_t in alpha_es
                for ratio in class_state_ratio]
conditions += [(1, 1, 1, 5)]

# ==== Loop Over Conditions ====
parser = argparse.ArgumentParser()
parser.add_argument("--i", type=int, required=True)
args = parser.parse_args()

task_idx = args.i
(alpha_e, alpha_t, ratio, num_states) = conditions[task_idx]

# ==== Load data ====
paths = load_paths_config()
datadir = paths["dataset_dir"] / "RandomHMMs"
data_filename = f"RandomHMM_"\
    + f"{num_states}_{alpha_e:0.2f}_{alpha_t:0.2f}_{ratio:0.1f}.pkl"
print(f"Data file: {data_filename}", flush=True)

with open(datadir / data_filename, "rb") as file:
    (hmm, hmm_params, task_params) = pkl.load(file)

training_data, validation_data, test_data = generate_hmm_dataset(
    hmm, hmm_params, dataset_params, task_params)

test_filename = (datadir / data_filename).with_stem(
    f"{(datadir / data_filename).stem}_test"
    )
with open(test_filename, "wb") as file:
    pkl.dump(test_data, file)

# ==== Training Parameters ====
model_specs = {
        'model_class': 'Linear',
        'in_size': task_params['num_classes'],
        'hidden_size': num_states,
        'out_size': task_params['num_classes'],
        'mlp_depth': 0,
        'num_blocks': 1
}

# ==== Train 5 independent seeds ====
for idx in range(5):
    seed = int(time.time())
    model_specs['key'] = jr.PRNGKey(seed)

    init_model = StackedLinearRNN(**model_specs)

    # == Saving parameters ==
    savedir = paths['rnn_data_dir'] / task_params['task']
    savedir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = savedir / (f"{model_specs['model_class']}_"\
        + f"{data_filename}_" \
        + f"{dataset_params['num_timesteps']}_{idx}")
    print(f"Model file: {filename}", flush=True)
    
    model, training_loss_history, validation_loss_history = \
            train_prediction_model(
            init_model, training_data, validation_data, 
            task_params, opt_params, model_specs, filename)

