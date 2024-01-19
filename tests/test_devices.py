#%%
import jax
import jax.numpy as jnp

#%%
a = jnp.zeros([100, 100])
print(jax.devices('gpu'))

b = a * a * a * a

#%%
print(b.devices())

