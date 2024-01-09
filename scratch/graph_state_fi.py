#%%
import tensorcircuit as tc
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from queso.sensors.tc.sensor import Sensor
from queso.configs import Configuration
from queso.io import IO

#%%
bundles = [[0, 1, 2], [3, 4, 5]]
nodes = [node for bundle in bundles for node in bundle]
g = nx.Graph()
g.add_nodes_from(nodes)

#%%
edges = []
for i, bundle_i in enumerate(bundles):
    for j, bundle_j in enumerate(bundles):
        if j >= i:
            continue
        for node_a in bundle_i:
            for node_b in bundle_j:
                edges.append([node_a, node_b])
print(edges)
g.add_edges_from(edges)

#%%
colors = ["red", "blue", "green"]
color_map = []
for i, bundle in enumerate(bundles):
    print(i, bundle)
    for node in bundle:
        color_map.append(colors[i])

nx.draw(g, node_color=color_map, with_labels=True)
plt.show()

#%%
io = IO(folder="photonic_graph_state", include_date=True)
config = Configuration()
config.n = len(g.nodes)
config.k = 4
config.seed = 122344

folder = f"2024-01-08_ansatz_tests"

# config.preparation = 'hardware_efficient_ansatz'
# config.preparation = 'brick_wall_cr'
# config.preparation = 'trapped_ion_ansatz'
config.preparation = 'photonic_graph_state_ansatz'
config.interaction = 'local_rx'
config.detection = 'local_r'
config.loss_fi = "loss_cfi"
config.graph_state = g

key = jax.random.PRNGKey(config.seed)

kwargs = dict(
    preparation=config.preparation,
    interaction=config.interaction,
    detection=config.detection,
    backend=config.backend,
    graph_state=config.graph_state
)


sensor = Sensor(config.n, config.k, **kwargs)
phi = sensor.phi
# theta, mu = sensor.theta, sensor.mu
theta = jax.random.uniform(key, shape=sensor.theta.shape)
mu = jax.random.uniform(key, shape=sensor.mu.shape)

sensor.circuit(theta, phi, mu).draw()
print(f"State vector: {sensor.state(theta, phi)}")
print(f"QFI: {sensor.qfi(theta, phi)} | CFI {sensor.cfi(theta, phi, mu)}")