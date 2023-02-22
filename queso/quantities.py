import tensorcircuit as tc


def quantum_fisher_information(circ, theta, phi, n, k):
    psi = circ(theta, phi, n, k).state()[:, None]

    f_dpsi_phi = tc.backend.jacrev(
        lambda _phi: circ(params=theta, phi=_phi, n=n, k=k).state()
    )
    d_psi = f_dpsi_phi(phi)

    fi = 4 * tc.backend.real(
        (tc.backend.conj(d_psi.T) @ d_psi) + (tc.backend.conj(d_psi.T) @ psi) ** 2
    ).squeeze()
    return -fi


def classical_fisher_information(circ, theta, phi, gamma, n, k):
    pr = circ(theta, phi, gamma, n, k).probability()

    dpr_phi = tc.backend.jacrev(
        lambda _phi: circ(theta, _phi, gamma, n, k).probability()
    )
    d_pr = dpr_phi(phi).squeeze()
    fi = tc.backend.sum(d_pr * d_pr / pr)
    return -fi
