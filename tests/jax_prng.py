import jax
import time

rng_key = jax.random.PRNGKey(time.time_ns())
rng_key, rng_subkey = jax.random.split(rng_key)
param = jax.random.uniform(
    key=rng_key,
    shape=[4],
    minval=0,
    maxval=1,
)
print(param)