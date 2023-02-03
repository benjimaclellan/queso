import tensorcircuit as tc
import jax.numpy as np
import optax

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”

#%%
params = backend.implicit_randn([3])


def func(a, b, c):
    dmc = tc.Circuit(1)
    dmc.r(0, theta=a, alpha=b, phi=c)
    return dmc.state()


df = backend.jacrev(func, argnums=(0, 1))

print(df(1.0, 1.0, 1.0))
out = df(1.0, 1.0, 1.0)