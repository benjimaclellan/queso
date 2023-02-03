import tensorcircuit as tc
import tqdm
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import matplotlib.pyplot as plt
import time

#%%
backend = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”
# tc.set_contractor("auto")  # “auto”, “greedy”, “branch”, “plain”, “tng”, “custom”


if __name__ == "__main__":
    n = 6
    k = 4


    def sensor(params, phi, gamma):
        dmc = tc.DMCircuit(n)

        for i in range(k):
            for j in range(n):
                dmc.r(j, theta=params[3 * j, i], alpha=params[3 * j + 1, i], phi=params[3 * j + 2, i])

            for j in range(1, n):
                dmc.cnot(j - 1, j)

            for j in range(n):
                dmc.phasedamping(j, gamma=gamma[0])
                # dmc.depolarizing(j, px=gamma[0], py=gamma[0], pz=gamma[0])

        for j in range(n):
            dmc.r(j, theta=params[3 * j, i], alpha=params[3 * j + 1, i], phi=params[3 * j + 2, i])

        # interaction
        for j in range(n):
            dmc.rz(j, theta=phi[0])

        # measurement
        for j in range(n):
            dmc.u(j, theta=params[3 * j, -1], phi=params[3 * j + 1, -1])

        return dmc


    phi = np.array([1.12314])
    gamma = np.array([0.0])
    params = backend.implicit_randn([3 * n, k + 1])

    dmc = sensor(params, phi, gamma)


    def cfi(_params, _phi, _gamma):
        def probs(_params, _phi, _gamma):
            return backend.abs(backend.diagonal(sensor(_params, _phi, _gamma).densitymatrix()))

        pr = probs(_params, _phi, _gamma)
        dpr_phi = backend.jacrev(lambda _phi: probs(_params=_params, _phi=_phi, _gamma=_gamma))
        d_pr = dpr_phi(phi).squeeze()
        fim = backend.sum(d_pr * d_pr / pr)
        return fim

    print(cfi(params, phi, gamma))

    def neg_cfi(_params, _phi, _gamma):
        return -cfi(_params, _phi, _gamma)
    #%%
    dmc.draw(output="text")
    #%%
    # cfi_val_grad_jit = backend.jit(backend.value_and_grad(lambda params: -cfi(_params=params, _phi=phi)))
    cfi_val_grad_jit = backend.jit(backend.value_and_grad(neg_cfi, argnums=0))

    val, grad = cfi_val_grad_jit(params, phi, gamma)
    print(val, grad)
    #%%
    opt = tc.backend.optimizer(optax.adagrad(learning_rate=0.2))
    params = backend.implicit_randn([3 * n, k + 1])

    for i in range(250):
        val, grad = cfi_val_grad_jit(params, phi, gamma)
        # print(grad)
        params = opt.update(grad, params)
        print(f"Step {i} | CFI {val}")
        # print(params)
    #%%

    def optimal_information_under_dephasing(gamma, progress=True):
        opt = tc.backend.optimizer(optax.adagrad(learning_rate=0.2))
        params = backend.implicit_randn([3 * n, k + 1])

        n_steps = 150
        loss = []
        for step in (pbar := tqdm.tqdm(range(n_steps), disable=(not progress))):
            val, grad = cfi_val_grad_jit(params, phi, gamma)
            params = opt.update(grad, params)
            loss.append(-val)
            if progress:
                pbar.set_description(f"Cost: {-val:.10f}")

        return val, np.array(loss)

    #%%
    df = []
    gammas = np.exp(np.linspace(-5, -1, 11))
    gammas = np.hstack([np.array([0.0]), gammas])
    t0 = time.time()

    for gamma in gammas:
        print(gamma)

        _vals_tmp = np.zeros(7)
        _loss = []
        for j in range(7):
            val, loss = optimal_information_under_dephasing(np.array([gamma]))

            _vals_tmp[j] = -val

        df.append(dict(
            gamma=gamma,
            cfi=np.mean(_vals_tmp),
            cfi_max=np.max(_vals_tmp),
            cfi_min=np.min(_vals_tmp),
            cfi_std=np.std(_vals_tmp),
            loss=loss,
        ))
    t = time.time() - t0
    print(t)

    #%%
    df = pd.DataFrame(df)
    print(df)

    #%%
    fig, ax = plt.subplots()
    ax.plot(df.gamma, df.cfi_max, label="optimal CFI")
    ax.fill_between(df.gamma, df.cfi_min, df.cfi_max, alpha=0.25, label="CFI range")
    ax.legend()
    ax.set(xlabel="Dephasing coefficient, $\gamma$", ylabel="Classical Fischer Information")
    plt.show()

    #%%
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cmap = sns.color_palette("flare", as_cmap=True)

    for i, gamma in enumerate(gammas):
        dfi = df[df.gamma == gamma]
        ax.plot(dfi.loss.iloc[0], color=cmap(i/len(gammas)), label=f"$\gamma={gamma:1.4f}$")

    ax.set(xlabel="Optimization iteration", label="Classical Fischer Information")
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),)
    fig.tight_layout()
    plt.show()
