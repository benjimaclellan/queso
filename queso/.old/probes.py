import tensorcircuit as tc


def cnot_2local_ansatz(params, phi, n, k):
    c = tc.Circuit(n)
    for i in range(k):
        for j in range(n):
            c.r(
                j,
                theta=params[3 * j, i],
                alpha=params[3 * j + 1, i],
                phi=params[3 * j + 2, i],
            )

        for j in range(1, n, 2):
            c.cnot(j - 1, j)

        for j in range(2, n, 2):
            c.cnot(j - 1, j)

    # interaction
    for j in range(n):
        c.rz(j, theta=phi)
    return c


def trapped_ion_ansatz(params, phi, n, k):
    c = tc.Circuit(n)
    for i in range(k):
        for j in range(n):
            c.r(
                j,
                theta=params[4 * j, i],
                alpha=params[4 * j + 1, i],
                phi=params[4 * j + 2, i],
            )

        for j in range(1, n, 2):
            c.rxx(j - 1, j, theta=params[4 * j + 3, i])

        for j in range(2, n, 2):
            c.rxx(j - 1, j, theta=params[4 * j + 3, i])

    # interaction
    for j in range(n):
        c.rz(j, theta=phi)

    return c


def build(name, n, k):
    if name == "cnot_2local_ansatz":
        circuit, shape = cnot_2local_ansatz, [3 * n, k]
    elif name == "trapped_ion_ansatz":
        circuit, shape = trapped_ion_ansatz, [4 * n, k]
    else:
        raise NotImplementedError("Not a valid probe state name")
    return circuit, shape
