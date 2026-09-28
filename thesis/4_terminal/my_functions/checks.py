"""
Physics regression checks. Run after any change to the Hamiltonian / lead code:

    from my_functions.checks import run_all
    run_all()

Every number printed should be at round-off level (or match the stated expectation).
"""

import numpy as np
from numpy.linalg import inv
from dataclasses import replace

from .params import Params
from .junction import FourTerminalJunction
from .fast_phase_sweep import FastPhaseSweep
from .hamiltonians import onsite_block, Vx, Vy, lattice_hamiltonian, block_repeat, flat_site_idx, SY
from .transport import current, conductance, energy_grid

SOC = dict(alpha=0.3, beta=-0.12, Bz=0.3, Bxy=0.2, theta_z=0.4)   # alpha -/+ beta both nonzero


def _G_small(J, E, S, keys):
    A = (E + 1j * J.p.eta) * np.eye(J.dim) - J.H_C
    for k in keys:
        A[np.ix_(J.idx[k], J.idx[k])] -= S[k]
    return inv(A)


def partition_x(k=3, E=0.13):
    """Move k N-lead cells into the centre, re-attach the same Sigma_L/R further out -> G_C unchanged.
    (No barrier: every x-bond, including the lead/centre one, is Vx(t_n) = Vx(t_c) here.)"""
    p = Params(nx=4, ny=3, mu_n=0.8, mu_c=0.5, eta=1e-6, **SOC)
    J = FourTerminalJunction(p)
    S = {kk: v[0] for kk, v in J.self_energies(np.array([E])).items()}
    Gs = _G_small(J, E, S, 'LR')
    NX = p.nx + 2 * k
    H = lattice_hamiltonian(NX, p.ny, onsite_block(p.t_n, p.mu_n, alpha=p.alpha, beta=p.beta, twod=True),
                            Vx(p.t_n, p.alpha, p.beta), Vy(p.t_n, p.alpha, p.beta))
    cen = [ix + NX * iy for iy in range(p.ny) for ix in range(k, k + p.nx)]
    onsC = onsite_block(p.t_c, p.mu_c, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z, alpha=p.alpha, beta=p.beta, twod=True)
    for s in cen:
        H[4 * s:4 * s + 4, 4 * s:4 * s + 4] = onsC
    A = (E + 1j * p.eta) * np.eye(len(H)) - H
    iL = flat_site_idx([NX * iy for iy in range(p.ny)])
    iR = flat_site_idx([NX - 1 + NX * iy for iy in range(p.ny)])
    A[np.ix_(iL, iL)] -= S['L']
    A[np.ix_(iR, iR)] -= S['R']
    ic = flat_site_idx(cen)
    return np.max(np.abs(inv(A)[np.ix_(ic, ic)] - Gs))


def partition_y(sc_leads, k=3, E=0.05):
    """Same in y: move k SC-lead rows into the centre, re-attach Sigma_T/B further out."""
    p = Params(nx=4, ny=3, mu_c=0.5, mu_s=0.9, delta=0.3, phi=0.7, eta=1e-6, **SOC)
    J = FourTerminalJunction(p, sc_leads)
    S = {kk: v[0] for kk, v in J.self_energies(np.array([E])).items()}
    Gs = _G_small(J, E, S, 'TB')
    twod = sc_leads == 'ribbon'
    hx_sc = Vx(p.t_s, p.alpha, p.beta) if twod else None
    row = {lab: lattice_hamiltonian(p.nx, 1, onsite_block(p.t_s, p.mu_s, delta=p.delta, phi=ph, twod=twod), hop_x=hx_sc)
           for lab, ph in (('T', p.phi), ('B', 0.0))}
    row['C'] = lattice_hamiltonian(p.nx, 1, onsite_block(p.t_c, p.mu_c, Bz=p.Bz, Bxy=p.Bxy, theta_z=p.theta_z,
                                                         alpha=p.alpha, beta=p.beta, twod=True),
                                   hop_x=Vx(p.t_c, p.alpha, p.beta))
    labels = ['T'] * k + ['C'] * p.ny + ['B'] * k
    m, n = 4 * p.nx, len(labels)
    V = block_repeat(Vy(1.0, p.alpha, p.beta), p.nx)          # all y-bonds have t = 1 here
    H = np.zeros((n * m, n * m), dtype=complex)
    for r, lab in enumerate(labels):
        H[r * m:(r + 1) * m, r * m:(r + 1) * m] = row[lab]
        if r < n - 1:
            H[r * m:(r + 1) * m, (r + 1) * m:(r + 2) * m] = V
            H[(r + 1) * m:(r + 2) * m, r * m:(r + 1) * m] = V.conj().T
    A = (E + 1j * p.eta) * np.eye(n * m) - H
    A[:m, :m] -= S['T']
    A[-m:, -m:] -= S['B']
    ic = np.arange(k * m, (k + p.ny) * m)
    return np.max(np.abs(inv(A)[np.ix_(ic, ic)] - Gs))


def particle_hole_and_fast_sweep():
    """PHS: T_ee(E) = T_hh(-E) etc.;  FastPhaseSweep == direct solve at other phases.  With disorder."""
    rng = np.random.default_rng(3)
    p = Params(nx=7, ny=5, mu_n=1.2, mu_c=0.9, mu_s=1.1, delta=0.3, phi=0.9, eta=1e-7, **SOC)
    J = FourTerminalJunction(p)
    dis = rng.normal(0, 0.3, p.nx * p.ny)
    for s, w in enumerate(dis):
        J.H_C[4 * s:4 * s + 4, 4 * s:4 * s + 4] += w * np.diag([1, 1, -1, -1])
    E = np.linspace(-0.4, 0.4, 9)                      # symmetric: E[::-1] = -E
    ch = J.channels(E)
    phs = max(np.max(np.abs(ch[s][a] - ch[s][b][::-1]))
              for s in ch for a, b in (('ee', 'hh'), ('eh_cross', 'he_cross'), ('eh_local', 'he_local')))
    fps, dfs = FastPhaseSweep(J, E), 0.0
    for phi in (0.3, 2.0, -2.5):
        Jp = FourTerminalJunction(replace(p, phi=phi))
        Jp.H_C = J.H_C
        c1, c2 = Jp.channels(E), fps.channels_at_phi(phi)
        dfs = max(dfs, max(np.max(np.abs(c1[s][k] - c2[s][k])) for s in c1 for k in c1[s]))
    return phs, dfs


def hermiticity_and_phs_of_H():
    p = Params(nx=4, ny=3, mu_c=0.4, **SOC)
    H = FourTerminalJunction(p).H_C
    P = np.kron(np.eye(p.nx * p.ny), np.kron(SY, SY))       # tau_y sigma_y per site
    return np.max(np.abs(H - H.conj().T)), np.max(np.abs(P @ H.conj() @ P.conj().T + H))


def clean_wire():
    """Centre = N-lead material, SC leads cut off by hand: T quantised, G_LL = T."""
    p = Params(nx=6, ny=3, mu_n=2.2, mu_c=2.2, eta=1e-9, kT=0.01, **{**SOC, 'Bz': 0.0, 'Bxy': 0.0})
    E = energy_grid(0.1, p.kT, 801)
    J = FourTerminalJunction(p)
    for lead in (J.sc_top, J.sc_bot):          # test-only: remove the SC/centre bond
        lead.V_coupling = np.zeros_like(lead.V_coupling)
    ch = J.channels(E, side='left')
    G_loc, G_nl = conductance(ch, E, 0.0, 0.0, p.kT)
    return ch['ee'][len(E) // 2], G_loc['total'], G_nl['total']


def analytic_vs_fd():
    p = Params(nx=6, ny=2, mu_n=3., mu_c=3., mu_s=3., delta=0.3,
               alpha=0.2, Bz=0.2, eta=1e-6, kT=0.01)
    E = energy_grid(0.15, p.kT, 801)
    ch = FourTerminalJunction(p).channels(E, side='left')
    V, h = np.linspace(-0.12, 0.12, 5), 1e-6
    G_loc, G_nl = conductance(ch, E, V, 0.3 * V, p.kT)
    fd_loc = (current(ch, E, V + h, 0.3 * V, p.kT)['total'] - current(ch, E, V - h, 0.3 * V, p.kT)['total']) / (2 * h)
    fd_nl = (current(ch, E, V, 0.3 * V + h, p.kT)['total'] - current(ch, E, V, 0.3 * V - h, p.kT)['total']) / (2 * h)
    return np.max(np.abs(G_loc['total'] - fd_loc)), np.max(np.abs(G_nl['total'] - fd_nl))


def ldos_consistency():
    """LDOS: full-inverse path == column path == fast phase sweep, and it is non-negative."""
    p = Params(nx=5, ny=4, mu_n=1.2, mu_c=0.9, mu_s=1.1, delta=0.3, phi=0.9, eta=1e-4, **SOC)
    J = FourTerminalJunction(p)
    E = np.linspace(-0.4, 0.4, 7)
    full = J.ldos(E)                                           # (N_E, ny, nx)
    sites = [J.site(2, 1), J.site(0, 3), J.site(4, 0)]
    cols = J.ldos(E, sites)
    d_paths = max(np.max(np.abs(full[:, s // p.nx, s % p.nx] - cols[:, k])) for k, s in enumerate(sites))
    fps = FastPhaseSweep(J, E, probe_sites=sites)
    d_fast = 0.0
    for phi in (0.3, 2.0):
        d_fast = max(d_fast, np.max(np.abs(FourTerminalJunction(replace(p, phi=phi)).ldos(E, sites)
                                           - fps.ldos_at_phi(phi))))
    T, rho = J.channels_and_ldos(E, sites)
    T_ref = J.channels(E)
    d_comb = max(np.max(np.abs(rho - cols)),
                 max(np.max(np.abs(T[s][k] - T_ref[s][k])) for s in T for k in T[s]))
    return max(d_paths, d_comb), d_fast, full.min()


def run_all():
    herm, phs_H = hermiticity_and_phs_of_H()
    print(f"H_C Hermitian / particle-hole symmetric        : {herm:.1e} / {phs_H:.1e}")
    print(f"partition invariance, N leads (x)              : {partition_x():.1e}")
    print(f"partition invariance, SC ribbons (y)           : {partition_y('ribbon'):.1e}")
    print(f"partition invariance, SC chains (y)            : {partition_y('chain'):.1e}")
    phs, dfs = particle_hole_and_fast_sweep()
    print(f"T(E) vs T(-E) particle-hole relations          : {phs:.1e}")
    print(f"FastPhaseSweep vs direct solve                 : {dfs:.1e}")
    T0, Gl, Gn = clean_wire()
    print(f"clean SOC wire: T_ee(0), G_LL, G_LR            : {T0:.6f}, {Gl:.6f}, {Gn:.6f}   (expect 4, 4, -4)")
    print("analytic conductance vs finite differences     : {:.1e}, {:.1e}".format(*analytic_vs_fd()))
    d_paths, d_fast, rho_min = ldos_consistency()
    print(f"LDOS full inv. vs columns vs combined / fast   : {d_paths:.1e} / {d_fast:.1e}   (min LDOS {rho_min:.2e} >= 0)")
