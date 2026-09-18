#%%
import sys
sys.path.append('../../..')

from hmmrnn.load_trained_model import load_trained_model
from config.load_config import load_paths_config

import jax
import jax.numpy as jnp

import pickle as pkl
import matplotlib.pyplot as plt
from scipy.stats import entropy

paths = load_paths_config()

data_dir = paths["dataset_dir"] / "CliqueHMM"
rnn_dir = paths["rnn_data_dir"] / "CliqueHMM"

#%% Metrics
# === Perplexity ===
def perplexity(probs, tokens):
    next_token = tokens[:,1:]
    probs = probs[:,:-1]

    idx = jnp.expand_dims(next_token, axis=-1)
    next_token_probs = -jnp.log2(
        jnp.take_along_axis(probs, idx, -1).squeeze(-1)
    )

    bits_per_seq = jnp.mean(next_token_probs, 
                            axis=1,
                            dtype=jnp.float32)
    return 2**bits_per_seq

def plot_token_probs(model_probs, opt_probs, title=None):
    fig, axes = plt.subplots(4,4)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(opt_probs[:,i], 'k-')
        ax.plot(model_probs[:,i], 'r--')

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    plt.show()

def c(A):
    n_rows, n_cols = A.shape
    indices = jnp.array([(i,j,k,l) for i in range(n_rows) 
           for j in range(n_rows)
           for k in range(n_cols)
           for l in range(n_cols)])
    
    def f(A, x):
        i, j, k, l = x
        return (A[i,k] * A[j,l]) / (A[j,k] * A[i,l])

    tau = jnp.nanmin(jax.vmap(f, in_axes=[None, 0])(A, indices))
    return (1-jnp.sqrt(tau)) / (1+jnp.sqrt(tau))


#%% Linear RNN
# Need to redo 4, 8

for k in [4]:
    for n in [4]:
        num_classes = n*k

        print("=== ", k, n, " ===") 

        kl_div = jnp.zeros((4, 20, 1))
        c_T = jnp.zeros((4, 20, 1))
        c_E = jnp.zeros((4, 20, 1))

        # Load Data
        for idx_e in range(4):
            for idx_t in range(20):
                data_filename = f"CliqueHMM_{k}_{n}_{num_classes}_{idx_t}_{idx_e}_128_test.pkl"
                with open(data_dir / data_filename, "rb") as file:
                    test_data = pkl.load(file)

                sequences = test_data["sequences"]
                hmm, hmm_params = test_data["hmm"]

                # Model perplexity
                opt_probs = jax.vmap(hmm.filter, in_axes=[None, 0])(
                    hmm_params, sequences).filtered_probs
                T = hmm_params.transitions.transition_matrix
                E = hmm_params.emissions.probs[:,0,:]

                opt_next_token_dist = opt_probs @ T @ E
                # opt_perplexity = opt_perplexity.at[idx_e, idx_t].set(
                #     perplexity(opt_next_token_dist, sequences).mean())
                c_T = c_T.at[idx_e, idx_t].set(c(T))
                c_E = c_E.at[idx_e, idx_t].set(c(E))

                # loop over RNN fits
                for mdl_idx in range(3):
                    print((idx_e, idx_t), mdl_idx)
                    rnn_filename = f"Linear_CliqueHMM_{k}_{n}_{num_classes}_" \
                        + f"{idx_t}_{idx_e}_128_{mdl_idx}"
                    model, _ = load_trained_model(f"{rnn_dir}/{rnn_filename}")

                    # True perplexity
                    model_next_token_dist = jax.nn.softmax(
                        jax.vmap(model)(sequences)
                    )
                    
                    kl_div = kl_div.at[idx_e, idx_t, mdl_idx].set(
                        entropy(model_next_token_dist, 
                                opt_next_token_dist, axis=2).mean()
                    )

        avg_kl_div = jnp.mean(kl_div, axis=-1)
        std_kl_div = jnp.std(kl_div, axis=-1)

        fig = plt.figure()
        plt.errorbar(c_T[0], avg_kl_div[0], std_kl_div[0],
                    color='r', marker='o', label=f"c_E = {c_E[0,0]}")
        plt.errorbar(c_T[1], avg_kl_div[1], std_kl_div[1],
                color='b', marker='o', label=f"c_E = {c_E[1,0]}")
        plt.errorbar(c_T[2], avg_kl_div[2], std_kl_div[2],
                color='orange', marker='o', label=f"c_E = {c_E[2,0]}")
        plt.errorbar(c_T[3], avg_kl_div[3],std_kl_div[3],
                color='mediumorchid',marker='o', label=f"c_E = {c_E[3,0]}")

        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('c_T')
        plt.ylabel('D_KL(model || Bayes)')

        plt.legend()
        fig.savefig(f'../clique_hmm_{k}_{n}_2.pdf')

# %%
k = 4
n = 4

num_classes = n*k

mat = jnp.zeros((20, 128))

print("=== ", k, n, " ===") 

kl_div = jnp.zeros((4, 20, 1))
c_T = jnp.zeros((4, 20, 1))
c_E = jnp.zeros((4, 20, 1))

idx_e = 3
for idx_t in range(20):
    data_filename = f"CliqueHMM_{k}_{n}_{num_classes}_{idx_t}_{idx_e}_128_test.pkl"
    with open(data_dir / data_filename, "rb") as file:
        test_data = pkl.load(file)

    sequences = test_data["sequences"]
    hmm, hmm_params = test_data["hmm"]

    # Model perplexity
    opt_probs = jax.vmap(hmm.filter, in_axes=[None, 0])(
        hmm_params, sequences).filtered_probs
    T = hmm_params.transitions.transition_matrix
    E = hmm_params.emissions.probs[:,0,:]

    opt_next_token_dist = opt_probs @ T @ E
    # opt_perplexity = opt_perplexity.at[idx_e, idx_t].set(
    #     perplexity(opt_next_token_dist, sequences).mean())
    c_T = c_T.at[idx_e, idx_t].set(c(T))
    c_E = c_E.at[idx_e, idx_t].set(c(E))

    # loop over RNN fits
    mdl_idx = 0

    print((idx_e, idx_t), mdl_idx)
    rnn_filename = f"Linear_CliqueHMM_{k}_{n}_{num_classes}_" \
        + f"{idx_t}_{idx_e}_128_{mdl_idx}"
    model, _ = load_trained_model(f"{rnn_dir}/{rnn_filename}")

    # True perplexity
    model_next_token_dist = jax.nn.softmax(
        jax.vmap(model)(sequences)
    )

    dkl = entropy(model_next_token_dist, 
                opt_next_token_dist, axis=2)

    plt.plot(dkl.mean(0))

    mat = mat.at[idx_t].set(dkl.mean(0))
plt.show()

# %%

# %%

plt.plot(mat.T)
# %%
