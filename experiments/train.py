import time
import jax
import jax.numpy as jnp

from queso.io import IO
from train_circuit import train_circuit
from sample_circuit import sample_circuit
from train_nn import train_nn

#%%
n = 4
k = 4

io = IO(folder=f"circ-nonlocal-detection-n{n}-k{k}/", include_date=True, include_id=False)
io.path.mkdir(parents=True, exist_ok=True)

# %% train circuit settings
phi_range = (-jnp.pi/4, jnp.pi / 4)
n_phis = 100
n_steps = 20000
lr = 1e-3
key = jax.random.PRNGKey(time.time_ns())
progress = True
plot = True
# circ_kwargs = dict(preparation="local_r", interaction="local_rx", detection="local_r")
# circ_kwargs = dict(preparation="brick_wall_cr", interaction="rx", detection="local")
circ_kwargs = dict(preparation="local_r", interaction="local_rx", detection="brick_wall_cr")

#%%
if True:
    train_circuit(
        io=io,
        n=n,
        k=k,
        key=key,
        phi_range=phi_range,
        n_phis=n_phis,
        n_steps=n_steps,
        lr=lr,
        contractor="plain",
        progress=progress,
        plot=plot,
        **circ_kwargs,
    )

#%% sample circuit settings
n_shots = 5000
n_shots_test = 1000

#%%
if True:
    sample_circuit(
        io=io,
        n=n,
        k=k,
        key=key,
        phi_range=phi_range,
        n_phis=n_phis,
        n_shots=n_shots,
        n_shots_test=n_shots_test,
        plot=plot,
        **circ_kwargs,
    )

#%% train estimator settings
key = jax.random.PRNGKey(time.time_ns())
# key = jax.random.PRNGKey(0)

n_epochs = 100
batch_size = 100
n_grid = n_phis  # todo: make more general - not requiring matching training phis and grid
nn_dims = [32, 32, 32, n_grid]
lr = 1e-3
plot = True
progress = True
from_checkpoint = False

#%%
if True:
    train_nn(
        io=io,
        key=key,
        nn_dims=nn_dims,
        n_steps=n_steps,
        n_grid=n_grid,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        plot=plot,
        progress=progress,
        from_checkpoint=from_checkpoint,
    )

# %% todo: benchmark estimator

