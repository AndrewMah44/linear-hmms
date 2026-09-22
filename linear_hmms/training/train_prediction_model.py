import jax
import jax.random as jr
import jax.numpy as jnp
import optax

import equinox as eqx
from optax import adamw, softmax_cross_entropy_with_integer_labels
import pickle as pkl

import time 

def decay_mask(tree):
    return jax.tree_util.tree_map(
        lambda v: isinstance(v, jnp.ndarray) and v.ndim > 1,
        tree
    )

# ==================
# Training model
# ==================
def train_prediction_model(model, training_data, validation_data,
                           task_params, opt_params, model_specs, 
                           filename, GRAD_CLIP_NORM=100.):
    """
    General wrapper function for fitting model on next-observation prediction 
    task.
    """
    _, training_key = jr.split(model_specs['key'], 2)

    # If no recurrent weight regularization parameter is given, use None
    if 'weight_decay' not in opt_params:
        opt_params['weight_decay'] = 0

    # == Format training data ==
    train_x = jax.device_put(
        training_data['sequences'][:, :-1].astype(jnp.int32))
    train_y = jax.device_put(
        training_data['sequences'][:, 1:].astype(jnp.int32))

    validation_x = jax.device_put(
        validation_data['sequences'][:, :-1].astype(jnp.int32))
    validation_y = jax.device_put(
        validation_data['sequences'][:, 1:].astype(jnp.int32))

    # == Define Training Step == #
    @eqx.filter_jit
    def loss_func(model, obs, next_obs):
        logits = jax.vmap(model)(obs)
        return softmax_cross_entropy_with_integer_labels(
            logits, next_obs
        ).mean()

    @eqx.filter_jit
    def training_step(model, obs, next_obs, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_func)(model, 
                                                           obs, 
                                                           next_obs)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return loss, model, opt_state
        
    # Run a single epoch on training data
    @eqx.filter_jit
    def run_one_training_epoch(model, opt_state, key, 
                            train_x, train_y):
        
        model_params, model_static = eqx.partition(
            model, eqx.is_inexact_array
            )

        # Permute data into mini-batches
        batch_indices = jr.permutation(
            key, jnp.arange(train_x.shape[0])).reshape(
                -1, opt_params['batch_size'])
        
        # Function to scan over mini-batches
        def scan_step(carry, i):
            model_params, opt_state = carry

            scan_model = eqx.combine(model_params, model_static)

            idx = batch_indices[i]
            batch_x = train_x[idx]
            batch_y = train_y[idx]
            loss, scan_model, opt_state = training_step(
                scan_model, batch_x, batch_y, opt_state)
            
            new_params, _ = eqx.partition(scan_model, 
                                            eqx.is_inexact_array)
            return (new_params, opt_state), loss

        # Actually scan
        init_carry = (model_params, opt_state)
        xs = jnp.arange(batch_indices.shape[0])
        (final_params, opt_state), loss_history = jax.lax.scan(
            scan_step, init_carry, xs=xs)
        
        final_model = eqx.combine(final_params, model_static)
        return final_model, opt_state, loss_history

    # == Initalize model and optimizer ==
    mask = decay_mask(model)
    learning_rate = opt_params['learning_rate']
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM), # Apply gradient clipping first
        adamw(learning_rate, 
              weight_decay=opt_params['weight_decay'],
              mask=mask)   # Then apply Adam optimizer steps
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # == Run optimization == 
    counter = 0 
    training_loss_history = []
    training_loss_history = []

    # Otherwise run as normal
    counter_thresh = 3
    validation_loss_history = [loss_func(model, validation_x, validation_y)]

    print(f"Initial loss: {validation_loss_history[-1]:0.5f}", flush=True)

    training_start_time = time.time()
    while counter < counter_thresh:

        epoch_start_time = time.perf_counter()

        trained_model, opt_state, epoch_training_loss = run_one_training_epoch(
            model, opt_state, training_key, train_x, train_y
        )
        jax.block_until_ready((trained_model, opt_state, epoch_training_loss))

        dt = time.perf_counter() - epoch_start_time

        epoch_validation_loss = float(
            loss_func(trained_model, validation_x, validation_y)
        )

        # Store training
        training_loss_history.append(epoch_training_loss)

        # if validation loss fails to achieve new minimum:
        # 1. decrease learning rate by a factor of 0.5
        # 2. add one to counter (if counter reaches 3, stop optimizing)
        if epoch_validation_loss > min(validation_loss_history):
            learning_rate *= 0.5
            mask = decay_mask(model)
            optimizer = optax.chain(
                optax.clip_by_global_norm(GRAD_CLIP_NORM),
                adamw(learning_rate, 
                    weight_decay=opt_params['weight_decay'],
                    mask=mask)
            )
            opt_state = optimizer.init(eqx.filter(trained_model, 
                                                  eqx.is_inexact_array))

            counter += 1
        
        # If validation loss does achieve new minimum, reset counter
        else:
            counter = 0

        print(f"Epoch {len(validation_loss_history):02d} ({dt:0.2f} s): " \
            + f"{epoch_validation_loss} " \
            + f"(counter = {counter})", flush=True)

        validation_loss_history.append(epoch_validation_loss)
        model = trained_model
        _, training_key = jr.split(training_key, 2)

        training_end_time = time.time()

    dt = training_end_time - training_start_time
    print(f"Fit took {dt:0.2f} seconds " \
        + f"({dt / len(validation_loss_history):0.5f} " \
        + "second/epoch)", flush=True)
    
    # == Save ==
    # split model into params (arrays) and static (non-arrays / functions)
    params, _ = eqx.partition(model, eqx.is_inexact_array)

    # serialise and save learned params
    eqx.tree_serialise_leaves(f"{filename}.eqx", params)

    # metadata to reconstruct model later
    metadata = {
        'model_specs': model_specs,
        'opt_params': opt_params,
        'task_params': task_params
    }

    with open(f"{filename}.meta.pkl", "wb") as file:
        pkl.dump({"metadata": metadata, 
                  "training_loss_history": training_loss_history,
                  "validation_loss_history": validation_loss_history},
                  file)
        
    return model, training_loss_history, validation_loss_history