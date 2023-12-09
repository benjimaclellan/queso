import jax
import jax.numpy as np
from functools import partial


def func():
    return [1]


# @partial(jax.jit, static_argnums=(1,))
# @jax.jit
def softmax(x, t):
    print(t)
    return np.log10(x[func])


key = jax.random.PRNGKey(0)
x = {func: jax.random.uniform(key=key, shape=[100, 100])}
print(x)

t = ["str"]
softmax(x, t)
# softmax(x, ["str"])
softmax_jit = jax.jit(partial(softmax, t=t))
softmax_jit(x)
