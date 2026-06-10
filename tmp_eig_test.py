import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

ny = 100
a = 20e-9
hbar = 1.05e-34
e = 1.602e-19
m = 9.1e-31
muB = 5.78e-2

tunits = (1e3 / e) * (hbar**2 / (2 * 0.038 * m * a**2))
mu = 2 / tunits
alpha_raw = 10e-9
beta_raw = -10e-9
alpha = alpha_raw / (a * tunits)
beta = beta_raw / (a * tunits)


def hamiltonian_for_energybands(t, mu, alpha, beta, Bz, ky=0.0):
    N = 2 * ny
    H = lil_matrix((N, N), dtype=np.complex128)
    onsite = (4 * t - mu + (alpha**2 + beta**2) / 4 - 2 * t * np.cos(ky))
    for i in range(ny):
        H[2 * i, 2 * i] = onsite + Bz
        H[2 * i + 1, 2 * i + 1] = onsite - Bz

    for i in range(ny - 1):
        up_i = 2 * i
        dn_i = 2 * i + 1
        up_j = 2 * (i + 1)
        dn_j = 2 * (i + 1) + 1

        H[up_i, up_j] += -t
        H[up_j, up_i] += -t
        H[dn_i, dn_j] += -t
        H[dn_j, dn_i] += -t

        soc = 1j * (beta + alpha) / 2
        H[up_i, dn_j] += soc
        H[dn_i, up_j] += soc
        H[dn_j, up_i] += -soc
        H[up_j, dn_i] += -soc

    phase = np.exp(1j * ky)
    soc_k = (alpha - beta) / 2
    for i in range(ny):
        up, dn = 2 * i, 2 * i + 1
        tk_up_dn = phase * (-soc_k)
        tk_dn_up = phase * (soc_k)
        tkc_up_dn = np.conj(tk_dn_up)
        tkc_dn_up = np.conj(tk_up_dn)

        H[up, dn] += tk_up_dn + tkc_up_dn
        H[dn, up] += tk_dn_up + tkc_dn_up

    H_dense = H.tocsr().toarray()
    return (H_dense + H_dense.conj().T) / 2

H = hamiltonian_for_energybands(1.0, mu, alpha, beta, 0.0, ky=np.pi / 3)
print('H shape', H.shape)
print('Hermitian check', np.max(np.abs(H - H.conj().T)))

for which in ['SM', 'LM', 'SA']:
    try:
        if which in ['SM', 'LM']:
            vals, _ = eigsh(csr_matrix(H), k=10, sigma=0.0, which=which)
        else:
            vals, _ = eigsh(csr_matrix(H), k=10, which=which)
        print(which, np.sort(vals)[:10])
    except Exception as e:
        print(which, 'error', type(e).__name__, e)

print('eigh first 10', eigh(H, eigvals=(0, 9))[0])
