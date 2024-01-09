import pytest

import jax
import time


@pytest.mark.parametrize(
    "n_samples",
    [
        2,
        4,
    ],
)
def test_jax_rng_generation(n_samples):
    rng_key = jax.random.PRNGKey(time.time_ns())
    rng_key, rng_subkey = jax.random.split(rng_key)
    param = jax.random.uniform(
        key=rng_key,
        shape=[4],
        minval=0,
        maxval=1,
    )
