import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time 
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
    'n_validation': 50_000,
    'n_test': 50_000,
    'num_timesteps': 128
}

opt_params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 64,
    'weight_decay': 1e-4
}

# ==== Task Parameters ====
idx_ts = range(20)
idx_es = range(4)
ks = [2, 4, 8]
ns = [2, 4, 8]

conditions = [(k, n, idx_t, idx_e) 
                for k in ks
                for n in ns
                for idx_t in idx_ts
                for idx_e in idx_es
                ]

# ==== Loop Over Conditions ====
num_clusters, state_per_cluster, idx_t, idx_e = conditions[0]

num_states = num_clusters * state_per_cluster
num_classes = num_states

# ==== Load data ====
paths = load_paths_config()
datadir = paths["dataset_dir"] / "ClusteredHMMs"
data_filename = f"ClusteredHMM_{num_clusters}_{state_per_cluster}"\
    + f"_{num_classes}_{idx_t}_{idx_e}.pkl"
print(f"Data file: {data_filename}", flush=True)

with open(datadir / data_filename, "rb") as file:
    (hmm, hmm_params, task_params) = pkl.load(file)

training_data, validation_data, _ = generate_hmm_dataset(
    hmm, hmm_params, dataset_params, task_params)

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
