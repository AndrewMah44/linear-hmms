#%% Imports
import jax
import pickle as pkl
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr
from statsmodels.stats.multitest import multipletests

from linear_hmms.utils.load_config import load_paths_config
from linear_hmms.utils.load_trained_model import load_trained_model

paths = load_paths_config()
datadir = paths['dataset_dir'] / "RandomHMM"
rnndir = paths['rnn_data_dir'] / "RandomHMM"

NUM_STATES = 16

#%% Functions
# === Perplexity ===
def filter_probs(hmm_pair, sequences):
    hmm, hmm_params = hmm_pair

    optimal_state_prob = jax.vmap(hmm.filter, in_axes=[None, 0])(
        hmm_params, sequences).filtered_probs
    T = hmm_params.transitions.transition_matrix
    E = hmm_params.emissions.probs[:,0,:]

    return optimal_state_prob @ T @ E

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

#%% Analysis: calculate c_T / c_E 
# == Set-up training == 
alpha_ts = [0.01, 0.25, 0.5, 0.75, 1]
alpha_es = [0.01, 0.25, 0.5, 0.75, 1]
alphas = [(alpha_e, alpha_t) 
          for alpha_e in alpha_es
          for alpha_t in alpha_ts]
ratios = [0.5, 1, 2]

c_t = jnp.zeros((3,25, 5))
c_e = jnp.zeros((3,25, 5))

for i, ratio in enumerate(ratios):
    for j, (alpha_e, alpha_t) in enumerate(alphas):
        print(i, j)
        data_filename = f"RandomHMM_{NUM_STATES}_" \
            + f"{alpha_e}_{alpha_t}_{ratio}_256_test.pkl"

        with open(datadir / data_filename, "rb") as file:
            test_data = pkl.load(file)

        _, hmm_params = test_data["hmm"]

        T = hmm_params.transitions.transition_matrix
        E = hmm_params.emissions.probs[:,0,:]

        c_t = c_t.at[i,j,:].set(c(T))
        c_e = c_e.at[i,j,:].set(c(E))

#%% Analysis: average KL-divergence
import time 
alpha_ts = [0.01, 0.25, 0.5, 0.75, 1]
alpha_es = [0.01, 0.25, 0.5, 0.75, 1]
alphas = [(alpha_e, alpha_t) 
          for alpha_e in alpha_es
          for alpha_t in alpha_ts]
ratios = [0.5, 1, 2]

kl_div = jnp.zeros((3, 25, 5))

total_start = time.time()
for i, ratio in enumerate(ratios):
    for j, (alpha_e, alpha_t) in enumerate(alphas):
        alpha_e, alpha_t = alphas[j]
        print(i, j)

        data_filename = f"RandomHMM_{NUM_STATES}_" \
            + f"{alpha_e}_{alpha_t}_{ratio}_256_test.pkl"

        with open(datadir / data_filename, "rb") as file:
            test_data = pkl.load(file)
        
        next_token_posterior = filter_probs(test_data["hmm"],
                                               test_data["sequences"])
        
        for idx in range(5):
            model_filename = f"Linear_RandomHMM_{NUM_STATES}_" \
                + f"{alpha_e}_{alpha_t}_{ratio}_256_{idx}"
            
            try:
                mdl, _ = load_trained_model(f"{rnndir}/{model_filename}")
            except:
                continue

            mdl_next_token_probs = jax.nn.softmax(
                jax.vmap(mdl)(test_data["sequences"])).block_until_ready()

            kl_div = kl_div.at[i, j, idx].set(
                entropy(mdl_next_token_probs, 
                        next_token_posterior, axis=2).mean()
            ).block_until_ready()

print(kl_div.mean())

#%% Analysis: linear decoder for belief states
# Data Conditions 
alpha_ts = [0.01, 0.25, 0.5, 0.75, 1]
alpha_es = [0.01, 0.25, 0.5, 0.75, 1]
alphas = [(alpha_e, alpha_t) 
          for alpha_e in alpha_es
          for alpha_t in alpha_ts]
ratios = [0.5, 1, 2]

decoded_kl_div = jnp.zeros((3, 25, 5))
random_kl_div = jnp.zeros((3, 25, 5))

for i, ratio in enumerate(ratios):
    ratio = ratios[i]

    for j, (alpha_e, alpha_t) in enumerate(alphas):
        print(i, j)

        data_filename = f"RandomHMM_16_{alpha_e}_{alpha_t}_{ratio}_256_test.pkl"

        with open(datadir / data_filename, "rb") as file:
            test_data = pkl.load(file)

        n = int(0.9 * test_data["sequences"].shape[0])
        test_sequences = test_data["sequences"][n:]

        hmm, hmm_params = test_data["hmm"]
        latent_state_posterior = jax.vmap(hmm.filter, in_axes=[None, 0])(
            hmm_params, test_sequences
        ).filtered_probs.reshape(-1,16)

        for idx in range(5):
            decoder_file = "posterior_decoder_" \
                +f"Linear_RandomHMM_16_{alpha_e}_{alpha_t}_{ratio}_256_{idx}"
            
            # Trained RNN
            try:
                with open(rnndir / (decoder_file + "_trained.pkl"), "rb") as file:
                    decoder_dict = pkl.load(file)
                trained_decoder = decoder_dict['decoder']
            except:
                print(decoder_file)
            model_filename = f"Linear_RandomHMM_{NUM_STATES}_" \
                + f"{alpha_e}_{alpha_t}_{ratio}_256_{idx}"
            mdl, specs = load_trained_model(f"{rnndir}/{model_filename}")

            hidden_activations = jax.vmap(mdl.blocks[0].rnn)(
                test_sequences).reshape(-1,16)
            decoded_posterior = jax.nn.softmax(
                jax.vmap(trained_decoder)(hidden_activations))

            decoded_kl_div = decoded_kl_div.at[i,j,idx].set(
                entropy(latent_state_posterior + 1e-32, 
                        decoded_posterior, axis=-1).mean() + 1e-32)
            
            # # Random RNN
            # with open(rnndir / (decoder_file + "_random.pkl"), "rb") as file:
            #     random_decoder_dict = pkl.load(file)
            # random_decoder = random_decoder_dict['decoder']

            # random_rnn = StackedLinearRNN(
            #     **specs['metadata']['model_specs']).blocks[0].rnn

            # random_activations = jax.vmap(random_rnn)(
            #     test_sequences).reshape(-1,16)
            # decoded_random_posterior = jax.nn.softmax(
            #     jax.vmap(random_decoder)(random_activations))

            # random_kl_div = random_kl_div.at[i,j,idx].set(
            #     entropy(latent_state_posterior, 
            #             decoded_random_posterior, axis=-1).mean())
            
print(jnp.mean(decoded_kl_div[2]), jnp.mean(random_kl_div[1])) 

# %% Analysis: local contraction tightness

import numpy as np

def hilbert_projective_distance(x, y, eps=1e-32):

    log_ratio = jnp.log(x + eps) - jnp.log(y + eps)
    return jnp.max(log_ratio) - jnp.min(log_ratio)

ratio = 1

contraction_ratio = jnp.zeros((25, 5))

for i, (alpha_e, alpha_t) in enumerate(alphas):
    print(i)

    # Load HMM
    data_filename = f"RandomHMM_16_{alpha_e}_{alpha_t}_{ratio}_256_test.pkl"

    with open(datadir / data_filename, "rb") as file:
        test_data = pkl.load(file)

    test_sequences = test_data["sequences"]

    hmm, hmm_params = test_data["hmm"]
    T = hmm_params.transitions.transition_matrix

    # Stationary Distribution
    evals, evecs = jnp.linalg.eig(T)
    idx = jnp.argsort(evals, descending=True)
    pi = evecs[:,idx[0]]

    c_T = c(T)

    for mdl_idx in range(5):
        model_filename = f"Linear_RandomHMM_{NUM_STATES}_" \
            + f"{alpha_e}_{alpha_t}_{ratio}_256_{mdl_idx}"
        mdl, specs = load_trained_model(f"{rnndir}/{model_filename}")

        hidden_activations = jax.vmap(mdl.blocks[0].rnn)(
            test_sequences).reshape(-1,16)
        decoded_posterior = jax.nn.softmax(
            jax.vmap(trained_decoder)(hidden_activations))

        numerator = jax.vmap(hilbert_projective_distance, 
                             in_axes=[0, None])(
            decoded_posterior @ T, pi)
        denominator = jax.vmap(hilbert_projective_distance, 
                               in_axes=[0, None])(
            decoded_posterior, pi)

        emp_contraction = jnp.mean(numerator / denominator)
        contraction_ratio = contraction_ratio.at[i,mdl_idx].set(
            emp_contraction / c_T
        )

    print(contraction_ratio)

fig, axes = plt.subplots()

im = axes.imshow(contraction_ratio.mean(1).reshape(5,5))
axes.set_xlabel('beta_T')
axes.set_ylabel('beta_E')

fig.colorbar(im, label='Sampled contraction ratio')

#%% Plot
from scipy.stats import mannwhitneyu

# Flatten everything for later
fig, axes = plt.subplots(1,3, figsize=(10, 3))

for idx in range(3):
    ces = c_e[idx].flatten()
    cts = c_t[idx].flatten()
    all_kl_div = kl_div[idx].flatten()

    goods = (cts < 1) & (ces < 1)

    axes[idx].errorbar([0, 1], 
                [jnp.mean(all_kl_div[goods]), jnp.mean(all_kl_div[~goods])],
                [jnp.std(all_kl_div[goods]) / jnp.sqrt(jnp.sum(goods)), 
                jnp.std(all_kl_div[~goods]) / jnp.sqrt(jnp.sum(~goods))],
                marker='o', linestyle='None')
    axes[idx].set_xticks([0, 1], ['c_E, c_T < 1', 'c_E = 1 or c_T = 1'])
    axes[idx].set_ylabel('log-KL div')
    axes[idx].set_yscale('log')

    stat, p = mannwhitneyu(all_kl_div[goods], all_kl_div[~goods], alternative='less')
    print(idx, stat, p)
    axes[idx].set_title(f"{idx}, p={p}")

fig.tight_layout()

fig.savefig(f'../error_by_cte_{NUM_STATES}_full.pdf')
plt.show()

#%%

# #%% Plotting: Error over Betas

# X_mean = jnp.mean(jnp.log10(kl_div), axis=2).reshape(3, 5, 5)

# # Median KL divergence over betas
# fig, axes = plt.subplots(1, 3, figsize=(12, 8))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     im = ax.imshow(X_mean[i])

#     cbar = fig.colorbar(im,fraction=0.046, pad=0.04)
#     cbar.set_label('mean log-KL-divergence')

#     ax.set_xticks(range(5))
#     ax.set_xticklabels(alpha_ts)
#     ax.set_xlabel('beta_t')

#     ax.set_yticks(range(5))
#     ax.set_yticklabels(alpha_es)
#     ax.set_ylabel('beta_e')

#     ax.set_title(f'M/N = {ratios[i]} ')
# fig.tight_layout()
# # fig.savefig(f'../error_by_HMM_{num_states}.pdf')

# #%%
# #%% Plotting: R^2 across HMMs

# print(f"corr KL-div and R^2: "\
#       + f"r={r_linear_decoding_v_perf.statistic:0.4f}, " \
#       + f"p={r_linear_decoding_v_perf.pvalue:0.4e}")

# # R^2 for each HMM
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# axes = axes.flatten()
# for i in range(3):
#     ax = axes[i]
#     im = ax.imshow(median_r_squared[i],
#                    cmap="viridis",
#                     interpolation="nearest"
#     )
#     im.set_rasterized(True)

#     cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
#     cbar.set_label('R^2')

#     ax.set_xticks(range(5))
#     ax.set_xticklabels(alpha_ts)
#     ax.set_xlabel('beta_t')

#     ax.set_yticks(range(5))
#     ax.set_yticklabels(alpha_es)
#     ax.set_ylabel('beta_e')

#     ax.set_title(f'Class / State ratio = {ratios[i]} \n ' \
#                  + f'{jnp.median(median_r_squared[i])}')
    
# # axes[3].scatter(median_kl_div.flatten(), median_r_squared.flatten())
# axes[3].set_xlabel('log KL-Div')
# axes[3].set_ylabel('R^2')
# axes[3].set_xscale('log')
# fig.tight_layout()
# plt.savefig("Fig_LinearDecodability.pdf")

# fig, ax = plt.subplots()
# im = ax.imshow(median_r_squared[0],
#                 cmap="viridis",
#                 interpolation="nearest"
# )
# im.set_rasterized(True)

# cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
# cbar.set_label('R^2')

# ax.set_xticks(range(5))
# ax.set_xticklabels([])

# ax.set_yticks(range(5))
# ax.set_yticklabels([])

# plt.savefig("heatmap.png", dpi=600, bbox_inches="tight")

# # %% Plotting: model performance over model sizes

# print(f"H statistic: {H1:0.4f}, {H2:0.4f}, {H3:0.4f}")
# print("Corrected p-values:", pvals_corr)
# print(f"Diff: {diff1:0.4f}, {diff2:0.4f}, {diff3:0.4f}")

# fig, axes = plt.subplots(3, figsize=(3, 5))
# for idx in range(3):
#     i = jnp.argmax(median_kl_div[idx])

#     data = jnp.vstack((kl_div[idx,i], kl_div_by_mdl_size[idx]))
#     axes[idx].errorbar(2**jnp.arange(4, 8), data.mean(1), data.std(1))
#     axes[idx].set_xticks(2**jnp.arange(4, 8))
#     axes[idx].set_title(ratios[idx])

# fig.supxlabel('Number of hidden units')
# fig.supylabel('KL-Divergence')

# fig.tight_layout()
# fig.savefig('../Rsquared_2.pdf')

# # %%
# def sci_latex(x, precision=3):
#     if x == 0:
#         return f"{0:.{precision}f}"
#     exp = int(jnp.floor(jnp.log10(abs(x))))
#     mant = x / 10**exp
#     return rf"{mant:.{precision}f} \times 10^{{{exp}}}"



# # %% Tables

# for (cond, row) in zip(alphas, kl_div[2]):
#     print(f"${cond}$ & " \
#           + " & ".join(f"${sci_latex(x)}$" for x in row) \
#           + f" & ${sci_latex(jnp.mean(row))}$"
#           + r" \\")



# #%%
# for idx in range(3):
#     i = jnp.argmax(median_kl_div[idx])

#     data = jnp.vstack((kl_div[idx,i], kl_div_by_mdl_size[idx]))
#     data = jnp.hstack((data, jnp.mean(data, axis=1, keepdims=True)))

#     print(alphas[i])
#     for i, row in enumerate(data):
#         print(f"{2**(4+i)} & " + " & ".join(f"${x:0.4f}$" for x in row) + r" \\")
    
#     print('')
    
# #%%


# for (cond, row) in zip(alphas, r_squared[2]):
#     print(f"${cond}$ & " \
#           + " & ".join(f"${x:0.4f}$" for x in row) \
#           + f" & ${row.mean():0.4f}$" \
#           + r" \\")


# # %%


