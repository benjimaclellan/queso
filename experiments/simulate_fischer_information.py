import tensorcircuit as tc
from jax import random

from queso.io import IO
from queso.quantities import quantum_fisher_information, classical_fisher_information
from queso import probes, sensors

backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


if __name__ == "__main__":
    ansatz = "trapped_ion_ansatz"
    n = 4
    k = 4
    seed = 0

    circ, shape = probes.build(ansatz, n, k)

    phi = 0.0
    # gamma = np.array([0.0])
    key = random.PRNGKey(seed)
    theta = random.uniform(key, shape)
    gamma = 0.0

    fisher_information = quantum_fisher_information
    # fisher_information = classical_fisher_information

    # %%
    fi_val_grad_jit = backend.jit(
        backend.value_and_grad(
            lambda _theta: fisher_information(circ=circ, theta=_theta, phi=phi, n=n, k=k),
            argnums=0,
        )
    )
    val, grad = fi_val_grad_jit(theta)
    print(val, grad)

