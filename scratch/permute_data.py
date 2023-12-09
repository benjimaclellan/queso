#%%
import jax
import jax.numpy as jnp

#%%
n_phis = 4
n_shots = 10
n = 4
shots = jnp.arange(n_shots * n_phis * n).reshape([n_phis, n_shots, n])
labels = n * jnp.arange(n_phis)

#%%
key = jax.random.PRNGKey(1)
keys = jax.random.split(key, 2)
perm_shots = jax.random.permutation(keys[0], shots, axis=0)
perm_labels = jax.random.permutation(keys[0], labels, axis=0)
sperm_shots = jax.random.permutation(keys[1], perm_shots, axis=1, independent=True)

keys = jax.random.split(key, n_phis)
sperm_shots = jnp.stack([jax.random.permutation(key, shots[i, :, :]) for i, key in enumerate(keys)])

jax.vmap(jax.random.permutation, in_axes=(0, 1), out_axes=(None, 1))(jax.random.split(key, shots.shape[1]), shots)