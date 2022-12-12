import jax
import jax.numpy as np
from jax.config import config

from qsense.functions import dagger, compile

config.update("jax_enable_x64", True)


def qfim(params, circuit, ket_i, keys):
    """
    Calculates the Quantum Fischer information matrix, with respect to the parameter keys provided in `keys`.
    :param params:
    :param circuit:
    :param keys:
    :return:
    """
    # parameters to compute QFI for must be complex datatype in order to automatically differentiate
    tunable_params = {key: np.complex128(params[key]) for key in keys}

    def fket(tunable_params):
        params.update(tunable_params)
        u = compile(params, circuit)
        return u @ ket_i

    fdket = jax.jacrev(fket, holomorphic=True)
    ket = fket(tunable_params)
    dket = fdket(tunable_params)

    # flatten parameters to a single list
    p = [(key, i) for key in keys for i in range(len(params[key]))]

    f = []
    for i, (key_a, a) in enumerate(p):
        fa = []
        for j, (key_b, b) in enumerate(p):
            da = dket[key_a][:, :, a]
            db = dket[key_b][:, :, b]
            f_ab = 4 * np.real(
                dagger(da) @ db - (dagger(da) @ ket) * (dagger(ket) @ db)
            )
            fa.append(f_ab.squeeze())
        f.append(fa)
    f = np.array(f)
    return f


def qfim_rho(params, circuit, ket_i, keys):
    # parameters to compute QFI for must be complex datatype in order to automatically differentiate
    tunable_params = {key: np.complex128(params[key]) for key in keys}

    def frho(tunable_params):
        params.update(tunable_params)
        u = compile(params, circuit)
        return u @ (ket_i @ dagger(ket_i)) @ dagger(u)

    def feigh(tunable_params):
        rho = frho(tunable_params)
        eigvals, eigvecs = np.linalg.eigh(rho)
        return np.complex128(eigvals), eigvecs

    fdrho = jax.jacrev(frho, holomorphic=True)
    fdeigh = jax.jacrev(feigh, holomorphic=True)

    drho = fdrho(tunable_params)
    eigvals, eigvecs = np.linalg.eigh(frho(tunable_params))
    deigvals, deigvecs = fdeigh(tunable_params)
    # print(drho)
    # print(deigvals)
    # flatten parameters to a single list
    p = [(key, i) for key in keys for i in range(len(params[key]))]
    # print(deigvals)
    # f = [[1]]
    f = []
    for key_a, a in p:
        fa = []
        for key_b, b in p:
            f_ab = 0.0
            for i in range(len(eigvals)):
                if eigvals[i] < 1e-9:
                    continue
                deval_ai = deigvals[key_a][i, a]
                deval_bi = deigvals[key_b][i, b]
                devec_ai = deigvecs[key_a][:, None, i, a]
                devec_bi = deigvecs[key_b][:, None, i, b]

                f_ab += deval_ai * deval_bi + 4 * eigvals[i] * np.real(dagger(devec_ai) @ devec_bi)
                print("f_ab", f_ab)
                # print(deval_ai.shape, deval_bi, devec_ai)

                for j in range(len(eigvals)):
                    if eigvals[j] < 1e-9:
                        continue
                    print(dagger(devec_ai).shape, eigvecs[:, None,  j].shape)
                    t = -8 * eigvals[i] * eigvals[j] / (eigvals[i] + eigvals[j])
                    print("t", t)
                    t *= np.real((dagger(devec_ai) @ eigvecs[:, None, j]) * (dagger(eigvecs[:, None, j]) @ devec_bi))
                    f_ab += t
                    print(f_ab)
            fa.append(f_ab.squeeze())
        f.append(fa)
    f = np.array(f)
    return f
