import jax
from jax import numpy as np

from qsense.sensor.functions import compile, dagger


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
        # return eigvals, eigvecs
        return np.complex128(eigvals), eigvecs

    # fdrho = jax.jacrev(frho, holomorphic=True)
    fdeigh = jax.jacrev(feigh, holomorphic=True)

    # drho = fdrho(tunable_params)
    eigvals, eigvecs = feigh(tunable_params)
    deigvals, deigvecs = fdeigh(tunable_params)
    print('deigvals', deigvals)
    print('eigvals', eigvals)

    # flatten parameters to a single list
    p = [(key, i) for key in keys for i in range(len(params[key]))]
    f = []
    for key_a, a in p:
        fa = []
        for key_b, b in p:
            f_ab = 0.0
            for i in range(len(eigvals)):
                if eigvals[i] < 1e-9:
                    print('skipping ', i)
                    continue
                deval_ai = deigvals[key_a][i, a]
                deval_bi = deigvals[key_b][i, b]
                devec_ai = deigvecs[key_a][:, i, None, a]
                devec_bi = deigvecs[key_b][:, i, None, b]
                print("total eigen sum", np.sum(np.abs(eigvecs[:, i])**2))
                print("deval_ai",deval_ai)
                print("real part shapes", dagger(devec_ai).shape, devec_bi.shape, np.real(dagger(devec_ai) @ devec_bi))
                term1 = deval_ai * deval_bi / eigvals[i]
                term2 = 4 * eigvals[i] * np.real(dagger(devec_ai) @ devec_bi)
                print("term1", term1)
                print("term2", term2)
                # f_ab += deval_ai * deval_bi / eigvals[i]
                f_ab += 4 * eigvals[i] * np.real(dagger(devec_ai) @ devec_bi)
                print("f_ab", f_ab)
                # print(deval_ai.shape, deval_bi, devec_ai)
                t = 0.0
                for j in range(len(eigvals)):
                    if eigvals[j] < 1e-9:
                        continue
                    print(dagger(devec_ai).shape, eigvecs[:, j, None].shape)
                    t += -8 * eigvals[i] * eigvals[j] / (eigvals[i] + eigvals[j]) * np.real((dagger(devec_ai) @ eigvecs[:, j, None]) * (dagger(eigvecs[:, j, None]) @ devec_bi))
                    print("t", t)
                    # t *= np.real((dagger(devec_ai) @ eigvecs[:, j, None]) * (dagger(eigvecs[:, j, None]) @ devec_bi))
                print(t)
                f_ab += t
                print(f_ab)
            fa.append(f_ab.squeeze())
        f.append(fa)

    # f = [[1]]

    f = np.array(f)
    return f


def cfim(params, probe_circ, interaction_circ, measure_circ, ket_i, keys):
    # tunable_params = {key: np.complex128(params[key]) for key in keys}
    tunable_params = {key: params[key] for key in keys}

    def fp(tunable_params):
        params.update(tunable_params)
        ket = ket_i
        ket = compile(params, probe_circ) @ ket
        ket = compile(params, interaction_circ) @ ket
        ket = compile(params, measure_circ) @ ket
        povm = np.abs(ket) ** 2
        return povm

    fdp = jax.jacrev(fp, holomorphic=False)
    p = fp(tunable_params)
    dp = fdp(tunable_params)

    # flatten parameters to a single list
    plist = [(key, i) for key in keys for i in range(len(params[key]))]

    f = []
    for i, (key_a, a) in enumerate(plist):
        fa = []
        for j, (key_b, b) in enumerate(plist):
            da = dp[key_a][:, 0, a]
            db = dp[key_b][:, 0, b]
            f_ab = np.sum(da * db / p[:, 0])
            fa.append(f_ab.squeeze())
        f.append(fa)
    f = np.array(f)
    return f