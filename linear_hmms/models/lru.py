import jax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp

# === Code for stacked Linear RNNs ===  
class LinearRNN(eqx.Module):
    Win: eqx.nn.Embedding
    Wrec: jax.Array
    in_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, *, init_scale=0.1, key):
        in_key, rec_key = jr.split(key, num=2)
        self.in_size = in_size
        self.hidden_size = hidden_size

        # Create embedding module
        self.Win = eqx.nn.Embedding(
            num_embeddings = self.in_size, 
            embedding_size = self.hidden_size,
            key = in_key)

        # Create recurrent connectivity matrix as a scaled orthogonal matrix
        self.Wrec = (init_scale / jnp.sqrt(hidden_size)) * jr.normal(
            key=rec_key, shape=(hidden_size, hidden_size))

    def __call__(self, inputs, key=None):
        """
        Run the linear RNN over the time dimension of inputs.

        inputs: array of shape [time_steps, in_size]
        Returns: array of hidden states shape [time_steps, hidden_size]
        """
        
        inputs_embedded = jax.vmap(self.Win)(inputs)

        def f(carry, inp):
            # Compute next hidden state
            h = self.Wrec @ carry + inp
            return h, h

        # Initialize carry to zeros with the correct hidden_size
        init_carry = jnp.zeros((self.hidden_size,))

        # Scan over time dimension
        _, hiddens = jax.lax.scan(f, init_carry, inputs_embedded)
        return hiddens

class ResidualLayer(eqx.Module):
    dimension: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, dimension, *, key):
        subkey1, subkey2 = jr.split(key)
        self.dimension = dimension
        self.linear1 = eqx.nn.Linear(dimension, dimension, key=subkey1)
        self.linear2 = eqx.nn.Linear(dimension, dimension, key=subkey2)
        self.layer_norm = eqx.nn.LayerNorm(dimension)

    def __call__(self, x, key=None):
        y = jax.nn.softplus(self.linear1(x))
        return self.layer_norm(self.linear2(y) + x)

class ResidualMLP(eqx.Module):
    width: int
    depth: int
    layers: eqx.Module

    def __init__(self, width, depth, *, key):
        self.width = width
        self.depth = depth

        if depth == 0:
            self.layers = eqx.nn.Identity()
        else:
            self.layers = eqx.nn.Sequential([
                ResidualLayer(width, key=k) for k in jr.split(key, num=depth)
            ])

    def __call__(self, x):
        return self.layers(x)

class LinearRecurrentBlock(eqx.Module):
    rnn: LinearRNN
    mlp: ResidualMLP
    hidden_size: int

    def __init__(self, in_size, hidden_size, mlp_depth, *, 
                 rnn_init_scale=0.1, key):
        rkey, mkey = jr.split(key)

        self.hidden_size = hidden_size
        self.rnn = LinearRNN(
            in_size=in_size,
            hidden_size=hidden_size,
            init_scale=rnn_init_scale,
            key=rkey
        )
        self.mlp = ResidualMLP(
            width=hidden_size,
            depth=mlp_depth,
            key=mkey
        )

    def __call__(self, x, key=None):
        x = self.rnn(x)
        return jax.vmap(self.mlp)(x)

class StackedLinearRNN(eqx.Module):
    blocks: eqx.nn.Sequential
    readout: eqx.nn.Linear

    def __init__(
            self, in_size, hidden_size, out_size, mlp_depth, num_blocks, 
            model_class, *, rnn_init_scale=0.1, key
        ):
        key1, key2, key3 = jr.split(key, num=3)
        blocks = [
            LinearRecurrentBlock(
                in_size, 
                hidden_size, 
                mlp_depth, 
                key=key1, 
                rnn_init_scale=rnn_init_scale
            )
        ]
        for k in jr.split(key2, num=(num_blocks - 1)):
            blocks.append(
                    LinearRecurrentBlock(
                    hidden_size, 
                    hidden_size, 
                    mlp_depth, 
                    key=k, 
                    rnn_init_scale=rnn_init_scale
                )
            )
        self.blocks = eqx.nn.Sequential(blocks)
        self.readout = eqx.nn.Linear(hidden_size, out_size, 
                                     key=key3)

    def __call__(self, x):
        return jax.vmap(self.readout)(self.blocks(x))