#%%
import aim
from aim import Run, Figure, Image, Repo
import matplotlib.pyplot as plt
import numpy as np

# %%
# Initialize a new run
# run = Run(experiment='some-experiment')
run = Run(
    experiment='server-test-2',
    repo='aim://192.168.0.18:53800',
)

#%%
# Log run parameters
run["hparams"] = {
    "learning_rate": 0.001,
    "batch_size": 32,
}
# %%
# Log metrics
for i in range(1, 10):
    run.track(i, name='loss', step=i, context={"subset": "train"})
    run.track(i, name='acc', step=i, context={"subset": "train"})

    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 100)
    ax.plot(x, np.random.random(x.shape))
    # plt.close(fig)
    # fig.close()
    aim_figure = Image(fig)
    run.track(aim_figure, name='test_plt', step=i, context={'subset': 'train'})

# %%
run.close()

#%%
repo = Repo(
    'aim://192.168.0.18:53800',
)

#%%
query = "metric.name == 'loss'" # Example query

run = repo.get_run("e4e1c889ba534f3f9d132cde")
print(run.metrics().dataframe())
df = run.metrics().dataframe(include_run=False, include_context=False, include_params=False, include_props=False)

#%%
fig, ax = plt.subplots()
ax.plot(df)