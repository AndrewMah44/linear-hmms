import time
import pickle as pkl
import jax.numpy as jnp
import jax.random as jr
from dynamax.hidden_markov_model import CategoricalHMM

from linear_hmms.utils.load_config import load_paths_config

def generate_random_hmm(key, alphas, num_states, num_classes):
    """
    Generates random HMMs by sampling rows of E and T from Dirichlet 
    distributions.

    INPUTS:
    key         - jr.PRNGKey for reproducibility
    alphas      - ndarray of concentraion parameters for E and T, (2,1)
    num_states  - number of states, int
    num_classes - number of emission types, int

    RETURNS:
    hmm        - hmm object from Dynamax
    hmm_params - hmm_params object from Dynamax

    note: E and T being sampled rowwise is matter of convenience for Python. 
    Everything is equivalent to column-wise sampling described in paper.

    note: For T, enforce that each state is reachable by at least one other 
    state (does not enforce fully connected but I think that's okay)
    """

    # Key management
    e_key, t_key = jr.split(key, 2)

    # Initial state probs (uniform over states)
    initial_probs = jnp.ones(num_states) / num_states

    # Concentration parameters for E and T
    alpha_e = alphas[0] * jnp.ones(num_classes)
    alpha_t = alphas[1] * jnp.ones(num_states)

    # Generate emission matrix
    emission_probs = jr.dirichlet(
        key = e_key, alpha = alpha_e, shape = (num_states,)
    )

    # Generate transition matrix
    transition_matrix = jr.dirichlet(
        key = t_key, alpha = alpha_t, shape = (num_states,)
    )

    # ensures that each state can be reached by at least one other
    while jnp.min(transition_matrix.sum(0)) < 0.1:        
        _, t_key = jr.split(t_key, 2)
        transition_matrix = jr.dirichlet(
        key = t_key, alpha = alpha_t, shape = (num_states,)
        )

    # Construct the HMM
    hmm = CategoricalHMM(num_states, num_emissions, num_classes)

    # Initialize the parameters struct with known values
    hmm_params, _ = hmm.initialize(
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_probs=emission_probs.reshape(num_states, 1, num_classes))
    
    return hmm, hmm_params

if __name__== "__main__":
    # ==== Set Up Directory ====
    paths = load_paths_config()
    savedir = paths['dataset_dir'] / "RandomHMMs"
    savedir.mkdir(parents=True, exist_ok=True)

    # ==== Set-up Dataset Parameters ====
    # HMM Parameters
    alpha_ts = [0.01, 0.25, 0.5, 0.75, 1]
    alpha_es = [0.01, 0.25, 0.5, 0.75, 1]
    class_state_ratio = [0.5, 1, 2]
    num_states_vec = [16, 32, 64]

    # Number of emissions observed at each timestep, not to be confused with 
    # the number of emission types (called num_classes)
    num_emissions = 1   

    # ==== Loop Over Datasets ====
    conditions = [(alpha_e, alpha_t, ratio, num_states) 
                  for num_states in num_states_vec
                  for alpha_e in alpha_es
                  for alpha_t in alpha_es
                  for ratio in class_state_ratio]

    for (alpha_e, alpha_t, ratio, num_states) in conditions:
        num_classes = int(num_states * ratio)
        
        task_params = {
            'task': 'RandomHMM',
            'num_states': num_states,
            'num_classes': num_classes,
            'alphas': [alpha_e, alpha_t],       
            'key': jr.PRNGKey(int(time.time()))
        }

        filename = savedir / (f"{task_params['task']}_"\
            + f"{num_states}_{alpha_e:0.2f}_{alpha_t:0.2f}_{ratio:0.1f}.pkl")
        print(filename)

        # ==== Generate data ====
        hmm, hmm_params = generate_random_hmm(
            task_params['key'],
            task_params['alphas'], 
            task_params['num_states'], 
            task_params['num_classes'])
    
        with open(filename, "wb") as file:
            pkl.dump((hmm, hmm_params, task_params), file)
