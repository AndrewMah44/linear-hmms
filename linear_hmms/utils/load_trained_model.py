import jax
import pickle as pkl
import equinox as eqx
import jax.random as jr
from linear_hmms.models.lru import StackedLinearRNN


def load_trained_model(filename, replace_win=False):
    with open(filename + ".meta.pkl", "rb") as file:
        res = pkl.load(file)

    # rebuild the same architecture (functions come from code, not from the file)
    model_specs = res["metadata"]["model_specs"]
    assert model_specs["model_class"] == "Linear", \
        f"{model_specs["model_class"]} is not valid model class"

    reconstructed = StackedLinearRNN(**model_specs)

    # load params and combine
    loaded_params = eqx.tree_deserialise_leaves(
        f"{filename}.eqx", eqx.filter(reconstructed, eqx.is_inexact_array))
    
    model = eqx.combine(
        loaded_params, eqx.partition(reconstructed, eqx.is_inexact_array)[1])
    
    if replace_win:
        def mapper(x):
            if isinstance(x, eqx.nn.Embedding):
                # Create a Linear layer with the same weights
                linear = eqx.nn.Linear(
                    in_features=x.num_embeddings,
                    out_features=x.embedding_size,
                    use_bias=False,
                    key=jr.PRNGKey(0)
                    )
                linear = eqx.tree_at(lambda lin: lin.weight, 
                                     linear, 
                                     x.weight.T)
                return linear
            return x

        # Apply the mapper to every leaf in the pytree
        # Leaves that aren't Embedding will be returned unchanged
        model = jax.tree_util.tree_map(mapper, 
                    model, 
                    is_leaf=lambda x: isinstance(x, eqx.nn.Embedding) \
                        or not isinstance(x, (tuple, list, dict, eqx.Module)))
        
    return model, res
