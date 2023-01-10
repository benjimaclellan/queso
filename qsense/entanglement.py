from qsense.utils import *
import qutip as qt


def genuine_multipartite_entanglement(rho):
    assert isinstance(rho, qt.Qobj)
    if rho.type == "ket":
        rho = qt.ket2dm(rho)

    n = len(rho.dims[0])
    partitions = partition(list(range(n)))

    for i, (inds_a, inds_b) in enumerate(partitions):
        # for inds in (inds_a, inds_b):
        prho = qt.ptrace(rho, inds_a)
        entropy = qt.entropy_vn(prho)
        print(f"Partition {i} of {len(partitions)} | partial trace keeping {inds_a}, entropy of {entropy}.")

    return
