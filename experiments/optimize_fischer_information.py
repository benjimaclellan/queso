import jax
import uuid
from functools import partial

from qsense.functions import *
from qsense.functions import initialize, compile
from qsense.qfi import qfim


if __name__ == "__main__":
    n = 4  # number of particles
    d = 2  # local dimensions

    ket_i = nketx0(n)
    # ket_i = nket_ghz(n)

    circuit = []
    circuit.append([(h, str(uuid.uuid4())) for i in range(n)])
    circuit.append([(cnot, str(uuid.uuid4())) for i in range(0, n, 2)])
    circuit.append([(phase, "phase") for i in range(n)])

    params = initialize(circuit)
    params['phase'] = np.array([(0.0 / 4) * np.pi + 0j])

    compile = jax.jit(partial(compile, circuit=circuit))
    u = compile(params)
    print(u @ ket_i)

    keys = ["phase"]
    qfi = lambda params, circuit, ket_i, keys: qfim(params, circuit, ket_i, keys)[0, 0]
    loss = jax.jit(partial(qfi,  circuit=circuit, ket_i=ket_i, keys=keys))
    # loss = jax.jit(partial(qfim,  circuit=circuit, ket_i=ket_i, keys=keys))

    ell = loss(params)
    print(ell)

    for _ in range(10):
        params = initialize(circuit)
        print(params)
        u = compile(params)
        ell = loss(params)
        print(ell)
    #
    # learning_rate = 0.4
    # n_step = 10
    # progress = False
    # optimizer = optax.adagrad(learning_rate)
    # opt_state = optimizer.init(params)


    # losses = []
    # grad = jax.grad(loss)
    # # for step in (
    # #         pbar := tqdm.tqdm(range(self.n_step), disable=(not self.progress))
    # # ):
    # for step in range(n_step):
    #     ell = loss(params)
    #     gradient = grad(params)
    #     updates, opt_state = optimizer.update(gradient, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     losses.append(ell)
    #     if progress:
    #         # pbar.set_description(f"Cost: {cost:.10f}")
    #         pass
    #     else:
    #         print(step, ell, params)
    # print(losses)
    #
    # plt.plot(losses)
    # plt.show()
