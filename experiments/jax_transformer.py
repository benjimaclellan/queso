#%% https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
import os
import numpy as np
import math
import tqdm
import pandas as pd
import json
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import time
import jax
import jax.numpy as jnp
from jax import random
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import h5py
from itertools import cycle

print("Device:", jax.devices()[0])

main_rng = random.PRNGKey(42)

from queso.sensors.tc.sensor import sample_int2bin
from queso.io import IO


#%%
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention

#%%
seq_len, d_k = 3, 2
main_rng, rand1 = random.split(main_rng)
qkv = random.normal(rand1, (3, seq_len, d_k))
q, k, v = qkv[0], qkv[1], qkv[2]
values, attention = scaled_dot_product(q, k, v)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Values\n", values)
print("Attention\n", attention)


#%%
# Test MultiheadAttention implementation
# Example features as input
main_rng, x_rng = random.split(main_rng)
embed_dim = 1
seq_len = 2  # N qubits
batch_size = 11

x = random.normal(x_rng, (batch_size, seq_len, embed_dim))
# Create attention
mh_attn = MultiheadAttention(embed_dim=1, num_heads=1)
# Initialize parameters of attention with random key and inputs
main_rng, init_rng = random.split(main_rng)
params = mh_attn.init(init_rng, x)['params']
# Apply attention with parameters on the inputs
out, attn = mh_attn.apply({'params': params}, x)
print('Out', out.shape, 'Attention', attn.shape)

del mh_attn, params

#%%
class EncoderBlock(nn.Module):
    input_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim=self.input_dim,
                                            num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x

#%%
# # Example features as input
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (11, 2, 1))
# Create encoder block
encblock = EncoderBlock(input_dim=1, num_heads=1, dim_feedforward=512, dropout_prob=0.1)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init({'params': init_rng, 'dropout': dropout_init_rng}, x, train=True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out = encblock.apply({'params': params}, x, train=True, rngs={'dropout': dropout_apply_rng})
print('Out', out.shape)

del encblock, params

#%%
class TransformerEncoder(nn.Module):
    num_layers : int
    input_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


#%%
class TransformerEstimator(nn.Module):
    n_output : int

    num_layers : int
    input_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        self.transenc = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.input_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout_prob,
        )

        self.dense = nn.Dense(self.n_output)

    def __call__(self, x, mask=None, train=True):
        x = self.transenc(x)
        x = self.dense(x)
        return x


#%%
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (11, 2, 1))
# Create Transformer encoder
transenc = TransformerEncoder(num_layers=5,
                              input_dim=1,
                              num_heads=1,
                              dim_feedforward=256,
                              dropout_prob=0.15)
# Initialize parameters of transformer with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = transenc.init({'params': init_rng, 'dropout': dropout_init_rng}, x, train=True)['params']
# Apply transformer with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
# Instead of passing params and rngs every time to a function call, we can bind them to the module
binded_mod = transenc.bind({'params': params}, rngs={'dropout': dropout_apply_rng})
out = binded_mod(x, train=True)
print('Out', out.shape)
attn_maps = binded_mod.get_attention_maps(x, train=True)
print('Attention maps', len(attn_maps), attn_maps[0].shape)

del transenc, binded_mod, params

#%%
lr = 1e-3

io = IO(folder="2023-07-21_estimator-performance-n2-k2")
hf = h5py.File(io.path.joinpath("samples.h5"), "r")
shots = jnp.array(hf.get("shots"))
counts = jnp.array(hf.get("counts"))
shots_test = jnp.array(hf.get("shots_test"))
probs = jnp.array(hf.get("probs"))
phis = jnp.array(hf.get("phis"))
hf.close()

n_phis = shots.shape[0]
n_shots = shots.shape[1]
n = shots.shape[2]

n_output = n_phis

#%%
phi_range = (jnp.min(phis), jnp.max(phis))

x = shots[:, :, :, None]
# x_init = x[:5, :4, :, :].reshape([ 5 * 4, 2, 1])
x_init = x[0, :11, :, :]
#%%
n_grid = n_phis
index = jnp.arange(n_grid)
assert n_phis == n_grid

labels = jax.nn.one_hot(index, num_classes=n_grid)
y = labels
print(index)
print(labels.sum(axis=0))

#%%
estimator = TransformerEstimator(
    n_output=n_output,
    num_layers=5,
    input_dim=n,
    num_heads=1,
    dim_feedforward=256,
    dropout_prob=0.15
)

# x_init = x_init.reshape([x_init.shape[0] * x_init.shape[1], n, 1])
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = estimator.init({'params': init_rng, 'dropout': dropout_init_rng}, x_init, train=True)['params']
main_rng, dropout_apply_rng = random.split(main_rng)
binded_mod = estimator.bind({'params': params}, rngs={'dropout': dropout_apply_rng})

#%%
out = binded_mod(x_init, train=True)
print(out.shape)

#%%
model = TransformerEstimator(
    n_output=n_output,
    num_layers=1,
    input_dim=1,
    num_heads=1,
    dim_feedforward=256,
    dropout_prob=0.0#15
)
print(model.tabulate(jax.random.PRNGKey(0), x_init, ))

main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, x_init, train=True)['params']
main_rng, dropout_apply_rng = random.split(main_rng)
binded_mod = model.bind({'params': params}, rngs={'dropout': dropout_apply_rng})
out = binded_mod(x_init, train=True)

#%%
@jax.jit
def train_step(state, batch):
    x_batch, y_batch = batch
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x_batch)
        loss = -jnp.sum(y_batch[:, None, :] * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean(axis=(0, 1))
        return loss

    loss_val_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = loss_val_grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss


#%%
def create_train_state(model, init_key, x, learning_rate):
    params = model.init(init_key, x)['params']
    print(f"Random initialization of parameters")

    schedule = optax.constant_schedule(lr)
    tx = optax.adam(learning_rate=schedule)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state

init_key = jax.random.PRNGKey(time.time_ns())
state = create_train_state(model, init_key, x_init, learning_rate=lr)

#%%
key = jax.random.PRNGKey(0)
n_epochs = 4
batch_size = 50
n_batches = n_shots // batch_size
progress = True

keys = jax.random.split(key, (n_epochs))
metrics = []
pbar = tqdm.tqdm(total=n_epochs * n_batches, disable=(not progress), mininterval=0.333)
for i in range(n_epochs):
    # shuffle shots
    # subkeys = jax.random.split(keys[i], n_phis)
    # x = jnp.stack([jax.random.permutation(subkey, x[k, :, :]) for k, subkey in enumerate(subkeys)])

    for j in range(n_batches):
        # x_batch = x[:, j * batch_size : (j + 1) * batch_size, :, :]
        x_batch = x[:, j * batch_size : (j + 1) * batch_size, :, :].reshape([n_phis * batch_size, n, 1])
        # y_batch = y  # use all phases each batch, but not all shots per phase
        y_batch = jnp.repeat(y, batch_size, axis=0)
        batch = (x_batch, y_batch)

        state, loss = train_step(state, batch)
        if progress:
            pbar.update()
            pbar.set_description(f"Epoch {i} | Batch {j} | Loss: {loss:.10f}", refresh=False)
        metrics.append(dict(step=i*n_batches + j, loss=loss))

pbar.close()
metrics = pd.DataFrame(metrics)

#%%
freqs = counts / counts.sum(axis=1, keepdims=True)
likelihood = freqs

#%%
fig, ax = plt.subplots()
ax.plot(metrics.step, metrics.loss)
ax.set(xlabel="Optimization step", ylabel="Loss")
fig.show()

#%%
xx = jnp.repeat(jnp.arange(n)[None, :], 7, axis=0)
binded_mod(xx, train=False)

#%%
bit_strings = sample_int2bin(jnp.arange(2 ** n), n)[:, :, None]
pred = binded_mod(bit_strings, train=False)
pred = jax.nn.softmax()
# pred = model.apply({'params': state.params}, bit_strings)
# pred = jax.nn.softmax(pred, axis=-1)

#%%
fig, axs = plt.subplots(nrows=3, figsize=[9, 6], sharex=True)
colors = sns.color_palette('deep', n_colors=bit_strings.shape[0])
markers = cycle(["o", "D", 's', "v", "^", "<", ">", ])
for i in range(bit_strings.shape[0]):
    ax = axs[0]
    xdata = jnp.linspace(phi_range[0], phi_range[1], pred.shape[1], endpoint=False)
    ax.plot(xdata,
            pred[i, :],
            ls='',
            marker=next(markers),
            color=colors[i],
            label=r"Pr($\phi_j | "+"b_{"+str(i)+"}$)")

    xdata = jnp.linspace(phi_range[0], phi_range[1], probs.shape[0], endpoint=False)
    axs[1].plot(xdata, probs[:, i], color=colors[i])
    axs[2].plot(xdata, freqs[:, i], color=colors[i], ls='--', alpha=0.3)

axs[-1].set(xlabel=r"$\phi_j$")
axs[0].set(ylabel=r"Posterior distribution, Pr($\phi_j | b_i$)")
fig.show()
