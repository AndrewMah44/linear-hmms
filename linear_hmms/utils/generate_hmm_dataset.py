import jax
import pickle as pkl
import jax.random as jr

def generate_hmm_dataset(hmm, hmm_params, dataset_params, task_params):
    train_key, validation_key, test_key = jr.split(
        task_params['key'], 3)
    
    # ==== Generate Training Dataset == #
    print("Making training data...")
    train_data = {}
    
    train_data['states'], train_data['sequences'], = jax.vmap(
        hmm.sample, in_axes=[None, 0, None])(
        hmm_params, 
        jr.split(train_key, dataset_params['n_train']), 
        dataset_params['num_timesteps']
    )
    train_data['sequences'] = train_data['sequences'][:, :, 0]
    
    # ==== Generate validation Dataset == #  
    print("Making validation data...")

    validation_data = {}
    
    validation_data['states'], validation_data['sequences'], = \
        jax.vmap(
        hmm.sample, in_axes=[None, 0, None])(
        hmm_params, 
        jr.split(validation_key, dataset_params['n_validation']), 
        dataset_params['num_timesteps']
    )
    validation_data['sequences'] = validation_data['sequences'][:,:,0]

    # ==== Generate test Dataset == #  
    print("Making test data...\n")

    test_data = {}

    test_data['states'], test_data['sequences'], = \
        jax.vmap(
        hmm.sample, in_axes=[None, 0, None])(
        hmm_params, 
        jr.split(test_key, dataset_params['n_test']), 
        dataset_params['num_timesteps']
    )
    test_data['sequences'] = test_data['sequences'][:,:,0]

    return train_data, validation_data, test_data
    # with open(
    #     filename.with_stem(f"{filename.stem}_train"), "wb"
    # ) as file:
    #     pkl.dump(train_data, file)

    # with open(
    #     filename.with_stem(f"{filename.stem}_validation"), "wb"
    # ) as file:
    #     pkl.dump(validation_data, file)
        
    # with open(
    #     filename.with_stem(f"{filename.stem}_test"), "wb"
    # ) as file:
    #     pkl.dump(test_data, file)
