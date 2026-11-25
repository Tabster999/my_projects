"""
Physics validation of the solvers AT THE THESIS JUNCTION, independent of any cached number.
Every test compares two different calculations, or one calculation with an exact law, so a passing run means the
numbers are right -- not only that they did not change.  (Regression against stored values: tests/test_regression.py.)

Run from the fourterminal folder:   python tests/test_physics.py        (~5 min on one thread)

Thesis junction: mixed SOC 16 meV nm at 0.15 pi, mu_C 0.7, mu_S 1, mu_N 3 meV, Delta 0.25 meV, a 10 nm, W 100 nm,
L 2 um, S/N bond 1, probe bond tau 0.5, field perpendicular to the SOC field.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'computations'))
import tools as st                                    # noqa: E402
from tools import myf                                 # noqa: E402

R = []


def check(name, ok, detail):
    R.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def thesis(tau=0.5, L=2000.0, **kw):
    """The thesis junction (as study_tuning.junction('mixed', 0.7, tau, L))."""
    alpha, beta = st.mixed_soc(16.0, 0.15 * np.pi)
    q = replace(st.case(100.0, L=L, alpha=alpha, beta=beta, a=10.0), tc=1.0, mu_C=0.7, tc_barr_transport=tau)
    return replace(q, **kw)


def s_matrix(rgf, phi, i_E=0):
    """Scattering matrix between the two probes from the Green's function (Fisher-Lee, lattice form):
    S_ab = -delta_ab + i Gamma_a^1/2 G_ab Gamma_b^1/2 on the open channels of each probe."""
    G_LL, G_LR, G_RL, G_RR = (g[i_E] for g in rgf.green_blocks(phi))
    W = []
    for Gam in (rgf.Gamma_L[i_E], rgf.Gamma_R[i_E]):
        w, U = np.linalg.eigh(Gam)
        keep = w > 1e-3 * w.max()                    # open channels; closed ones carry only Gamma ~ eta (~1e-7 t)
        W.append(U[:, keep] * np.sqrt(w[keep]))
    WL, WR = W
    S = np.block([[-np.eye(WL.shape[1]) + 1j * WL.conj().T @ G_LL @ WL, 1j * WL.conj().T @ G_LR @ WR],
                  [1j * WR.conj().T @ G_RL @ WL, -np.eye(WR.shape[1]) + 1j * WR.conj().T @ G_RR @ WR]])
    return S


def test_rgf_vs_dense():
    """Two different algorithms (RGF + Dyson vs a dense inverse of the whole centre), L shortened to 400 nm."""
    for W, seed in ((0.0, 0), (2.0, 3)):
        q = thesis(L=400.0, disorder=W, disorder_seed=seed)
        p = replace(q.params_transport(E_Z=0.72), phi=0.9 * np.pi)
        E = np.array([0.0, 0.02 / q.t, -0.05 / q.t])            # 0, 20, -50 ueV
        J = myf.FourTerminalJunction(p)
        dense = J.channels(E, 'left')
        rgf = myf.RGFFourTerminal(J, E).channels_at_phi(p.phi, 'left')
        d = max(np.abs(dense[k] - rgf[k]).max() for k in dense)
        check(f"RGF == dense (W {W} meV)", d < 1e-9, f"max |difference| over six channels, three energies {d:.1e}")


def unitarity_defect(q, E_Z, phi):
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=E_Z)), np.array([0.0]))
    S = s_matrix(rgf, phi)
    return np.abs(S @ S.conj().T - np.eye(len(S))).max(), len(S)


def test_unitarity():
    """Below the gap no quasiparticle enters the superconductors: S between the probes must be unitary -- up to the
    absorption eta adds (eta is an imaginary energy, i.e. a loss rate).  So the defect must (a) be small and (b) fall
    tenfold when eta does; a solver error would not scale with eta."""
    for W in (0.0, 2.0):
        q = thesis(disorder=W, disorder_seed=0)
        for E_Z, phi in ((0.72, np.pi), (0.30, 0.0), (1.06, 1.3 * np.pi)):
            d1, n = unitarity_defect(q, E_Z, phi)
            d2, _ = unitarity_defect(replace(q, eta_rel_transport=q.eta_rel_transport / 100), E_Z, phi)
            # measured 2026-11-25 at E_Z 0.72: 4.2e-4 / 5.8e-5 / 6.3e-6 / 6.1e-7 at eta x 1 / 0.1 / 0.01 / 0.001 -> no floor;
            # linear below the production eta, slightly sub-linear at it
            check(f"unitarity (W {W}, E_Z {E_Z}, phi {phi / np.pi:.1f} pi)", n == 8 and d1 < 1e-2 and d1 / d2 > 20,
                  f"{n} channels; |S S^+ - 1| {d1:.1e} at eta, {d2:.1e} at eta/100 (vanishes with eta: absorption only)")


def test_phs_and_kappa():
    """Particle-hole symmetry at E = 0: T_ee = T_hh, T_eh = T_he, so kappa = T_ee + T_he equals the symmetric form;
    and kappa = (1 - det r^2)/2 at a two-Majorana point."""
    q = thesis()
    rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=0.72)), np.array([0.0]))
    c = {k: float(v[0]) for k, v in rgf.channels_at_phi(np.pi, 'left').items()}
    d = max(abs(c['ee'] - c['hh']), abs(c['eh_cross'] - c['he_cross']))
    check("PHS at E = 0", d < 1e-9, f"|T_ee - T_hh|, |T_eh - T_he| <= {d:.1e}")
    k, dr = c['ee'] + c['he_cross'], st.det_r(rgf, np.pi)
    S = s_matrix(rgf, np.pi)
    n = S.shape[0] // 2
    dr_S = np.linalg.det(S[:n, :n])
    check("det r: channel code vs S-matrix block", abs(dr - dr_S.real) < 1e-6 and abs(dr_S.imag) < 1e-8,
          f"{dr:+.6f} vs {dr_S.real:+.6f} (imag {dr_S.imag:.1e})")
    check("kappa = (1 - det r^2)/2 at the resonance", abs(k - (1 - dr ** 2) / 2) < 0.01,
          f"kappa {k:.4f}, (1 - det r^2)/2 {(1 - dr ** 2) / 2:.4f}")


def test_exact_normal_limit():
    """Superconductors detached (S/N bond 0), field off, strip at the probe density, probe bond 1: the strip is a
    continuation of the probe, so T_ee = number of open spin channels exactly and nothing is Andreev reflected."""
    out = []
    for f in (1.0, 0.1):
        q = replace(thesis(tau=1.0), tc=0.0, mu_C=3.0)
        q = replace(q, eta_rel_transport=q.eta_rel_transport * f)
        rgf = myf.RGFFourTerminal(myf.FourTerminalJunction(q.params_transport(E_Z=0.0)), np.array([0.0]))
        out.append({k: float(v[0]) for k, v in rgf.channels_at_phi(np.pi, 'left').items()})
    c, c2 = out
    # eta absorbs over the strip: 2 channels x 2 eta L / (hbar v), hbar v = 2 a sqrt(mu t) at the subband energy
    q = replace(thesis(tau=1.0), tc=0.0, mu_C=3.0)
    E1 = 2 * q.t * (1 - np.cos(np.pi / (q.n_across + 1)))
    loss = 2 * 2 * (q.eta_rel_transport * q.Delta) * q.L / (2 * q.a * np.sqrt((3.0 - E1) * q.t))
    check("ideal normal strip: T_ee = 2 up to eta absorption",
          abs((2 - c['ee']) / loss - 1) < 0.2 and 8 < (2 - c['ee']) / (2 - c2['ee']) < 12 and c['he_cross'] < 1e-9
          and c['eh_local'] < 1e-9,
          f"2 - T_ee = {2 - c['ee']:.2e} (absorption estimate {loss:.2e}), at eta/10 {2 - c2['ee']:.2e}; "
          f"T_he {c['he_cross']:.1e}, local Andreev {c['eh_local']:.1e}")


def test_poles_vs_transport():
    """Two different calculations of the same physics: det r from the RGF Green's function vs 1 - 2 G_L G_R / |l1 l2|
    from the poles of the open junction (tools.nh_spectrum), clean and disordered."""
    tau = 0.5
    for W, seed in ((0.0, 0), (2.0, 1)):
        q = thesis(disorder=W, disorder_seed=seed)
        dr = st.transport_point(q, 0.72, np.pi)[3]
        P = {c: st.nh_spectrum(q, 0.72, [np.pi], *c, k=8)[0][0] for c in ((0.0, 0.0), (tau, 0.0), (0.0, tau), (tau, tau))}
        eta0 = -P[(0.0, 0.0)][:2].imag.sum() / 2
        gL = -P[(tau, 0.0)][:2].imag.sum() - eta0
        gR = -P[(0.0, tau)][:2].imag.sum() - eta0
        d_pole = 1 - 2 * gL * gR / abs(P[(tau, tau)][0] * P[(tau, tau)][1])
        check(f"det r: poles vs RGF (W {W})", abs(d_pole - dr) < 0.05, f"{d_pole:+.4f} vs {dr:+.4f}")


def test_gamma_two_ways():
    """At an eps zero (E_Z 0.285, phi pi): the pole width equals the half width of the local Andreev line."""
    q = thesis()
    lam = st.nh_spectrum(q, 0.285, [np.pi], 0.5, 0.0, k=8)[0][0]
    g_pole = (-lam[:2].imag).max()                    # the broad pole of the pair (the other is the far Majorana)

    def andreev(E):
        c = dict(zip(st.CHANNELS, st.channel_point(q, 0.285, E)))
        return float(np.ravel((np.asarray(c['eh_local']) + np.asarray(c['he_local'])) / 2)[0])
    A0, lo, hi = andreev(0.0), 1e-5, 1e-1
    for _ in range(16):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if andreev(mid) > A0 / 2 else (lo, mid)
    g_line = np.sqrt(lo * hi) * 1e3
    check("Gamma: pole width = Andreev half width", abs(g_line / g_pole - 1) < 0.02 and A0 > 0.99,
          f"pole {g_pole:.3f} ueV, line {g_line:.3f} ueV, peak height {A0:.4f} (1 = 2e^2/h)")


if __name__ == "__main__":
    n0 = myf.leads.N_UNRELIABLE
    for f in (test_exact_normal_limit, test_phs_and_kappa, test_rgf_vs_dense, test_unitarity, test_poles_vs_transport,
              test_gamma_two_ways):
        f()
    check("no failed lead solves", myf.leads.N_UNRELIABLE == n0, f"{myf.leads.N_UNRELIABLE - n0} failed")
    print(f"\n{sum(R)} / {len(R)} passed", flush=True)
    sys.exit(0 if all(R) else 1)
