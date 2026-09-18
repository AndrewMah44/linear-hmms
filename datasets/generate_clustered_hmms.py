#%%
import time
import pickle as pkl
import jax.random as jr
import jax.numpy as jnp
from scipy.linalg import block_diag
from dynamax.hidden_markov_model import CategoricalHMM

from linear_hmms.utils.load_config import load_paths_config

def generate_clustered_hmm(
        keys, 
        eps_e, 
        eps_t, 
        num_cliques, 
        num_states_per_clique):

    """
    Generates random HMMs by sampling rows of E and T from Dirichlet 
    distributions.

    INPUTS:
    key                   - Array of keys for reproducibility. [2,1]
    eps_e                 - weighting between E_dense and E_sparse. float64
    eps_e                 - weighting between T_cluster and T_rand. float64
    num_cliques           - number of cliques. int
    num_states_per_clique - number of states in each cluster. int

    RETURNS:
    hmm        - hmm object from Dynamax
    hmm_params - hmm_params object from Dynamax

    note: E and T being sampled rowwise is matter of convenience for Python. 
    Everything is equivalent to column-wise sampling described in paper.
    """
    # Total number of states
    num_states = num_cliques * num_states_per_clique
    num_classes = num_states

    # Initial probabilities
    initial_probs = jnp.ones(num_states) / num_states

    # === Generate transition matrix ===
    clique_t_key, anticlique_t_key, _, _ \
        = jr.split(keys[0], 4)

    _, _, clique_e_key,anticlique_e_key \
        = jr.split(keys[1], 4)
     
    # Generate Clique-y transition matrix
    clique_ts = jr.dirichlet(
            key = clique_t_key,
            alpha = jnp.ones(num_states_per_clique),
            shape = (num_cliques, num_states_per_clique)
        )

    T_cluster = block_diag(*clique_ts)

    T_rand = jr.dirichlet(
            key = anticlique_t_key,
            alpha = jnp.ones(num_states),
            shape = (num_states,)
        )

    transition_matrix = eps_t * T_cluster + (1-eps_t) * T_rand

    # === Generate emission matrix ===
    E_sparse = jr.dirichlet(
            key = clique_e_key,
            alpha = 0.01*jnp.ones(num_classes),
            shape = (num_states,)
        )
    
    E_dense = jr.dirichlet(
            key = anticlique_e_key,
            alpha = jnp.ones(num_classes),
            shape = (num_states,)
        )
    
    emission_probs = eps_e * E_sparse + (1-eps_e) * E_dense
    
    # == Generate data == 
    hmm = CategoricalHMM(num_states, 1, num_classes)
    
    hmm_params, _ = hmm.initialize(
         initial_probs = initial_probs,
        transition_matrix = transition_matrix,
        emission_probs = emission_probs.reshape(num_states, 1, num_classes)
    )

    return hmm, hmm_params

if __name__=="__main__":
    # ==== Set Up Directory ====
    paths = load_paths_config()
    savedir = paths['dataset_dir'] / "ClusteredHMMs"
    savedir.mkdir(parents=True, exist_ok=True)

    # ==== Set-up Dataset Parameters ====
    # HMM Parameters
    eps_ts = jnp.linspace(0, 1, 20)
    eps_es = [0.01, 0.25, 0.75, 0.99]
    ks = [2, 4, 8]
    ns = [2, 4, 8]

    # hardcode key to make sure only difference between models is lambda
    keys = [jr.PRNGKey(1114), jr.PRNGKey(1116)]

    # ==== Loop Over Datasets ====
    conditions = [(k, n, eps_t, eps_e, idx_t, idx_e) 
                  for k in ks
                  for n in ns
                  for idx_e, eps_e in enumerate(eps_es)
                  for idx_t, eps_t in enumerate(eps_ts)
                  ]

    for (num_cliques, num_states_per_clique, 
         eps_t, eps_e, idx_t, idx_e) in conditions:
        num_classes = num_states_per_clique * num_cliques

        # ==== Set task parameters ====
        task_params = {
            'task': 'ClusteredHMM',
            'num_cliques': num_cliques,
            'num_states_per_clique': num_states_per_clique,
            'num_classes': num_classes,
            'lam': (eps_t, eps_e),
            'key': jr.PRNGKey(int(time.time()))
        }

        filename = savedir / (f"{task_params['task']}_"\
            + f"{num_cliques}_{num_states_per_clique}_{num_classes:02}_" \
            + f"{idx_t:02}_{idx_e:02}.pkl")
        print(filename)

        # ==== Generate data ====
        hmm, hmm_params = generate_clustered_hmm(
            keys, 
            eps_e, 
            eps_t, 
            num_cliques, 
            num_states_per_clique)

        with open(filename, "wb") as file:
            pkl.dump((hmm, hmm_params, task_params), file)
    