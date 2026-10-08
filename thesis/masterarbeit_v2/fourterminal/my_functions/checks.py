"""
Physics and solver regression checks.  Run after any change to the Hamiltonian, lead or solver code:

    from my_functions.checks import run_all
    run_all()                      # ~1 minute; every line should say PASS

1. H_C is Hermitian and particle-hole symmetric (both models, SOC + Zeeman + per-region SOC)
2. RGF == dense junction.channels == FastPhaseSweep (Caroli transmissions, both leads, several phases)
3. particle-hole symmetry of the transmissions: T_ee(-E) = T_hh(E), T_he(-E) = T_eh(E)
3b. the same at E = 0 itself (T_ee = T_hh, T_eh = T_he): a free self-check that catches
    small-eta Sancho-Rubio breakdown which the lead residual test passes (see phs_residual)
4. clean normal wire: integer transmission, no Andreev processes
5. lead solver: eigenmode fallback == Sancho-Rubio where healthy; the known E = 0 failure is repaired
"""
from dataclasses import replace
import numpy as np

from .params import Params
from .junction import FourTerminalJunction
from .rgf import RGFFourTerminal, central_fingerprint
from .transport import phs_residual
from .compute import ldos
from .fast_phase_sweep import FastPhaseSweep
from . import leads

CASES = {
    'rashba': Params(model='rashba', nx=10, ny=6, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=0.3, t_s=1.0, mu_s=0.15, delta=0.35,
                     tc_top=1.1, tc_bot=0.9, tc_barr=1.0, alpha=1.2, beta=-0.2, Bz=0.1, Bxy=0.05, theta_z=0.4, Bz_s=0.94, eta=1e-5),
    'rashba, per-region SOC': Params(model='rashba', nx=10, ny=6, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=0.3, t_s=1.0, mu_s=0.15,
                     delta=0.35, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, alpha_c=0.6, alpha_n=0.0, Bz=0.1, Bz_n=0.0,
                     Bz_s=0.94, eta=1e-5),
    'dirac': Params(model='dirac', nx=10, ny=6, t_n=1.0, mu_n=0.5, t_c=1.0, mu_c=0.1, t_s=1.0, mu_s=0.77, delta=0.35,
                    tc_top=1.0, tc_bot=1.0, tc_barr=1.0, Bz_s=-0.3, m0=0.8, m0_c=0.8, m0_n=0.8, eta=1e-5),
}
E = np.array([-0.2, -0.03, 0.0, 0.04, 0.25])
_RESULTS = []


def _report(name, value, tol):
    ok = bool(np.isfinite(value) and value < tol)
    _RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:62s} {value:.1e}  (tol {tol:.0e})")
    return ok


def hermiticity_and_phs_of_H():
    U4 = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))      # tau_y sigma_y (then K)
    for name, p in CASES.items():
        H = FourTerminalJunction(replace(p, phi=1.1)).H_C
        U = np.kron(np.eye(H.shape[0] // 4), U4)
        _report(f"H_C Hermitian ({name})", np.abs(H - H.conj().T).max(), 1e-12)
        _report(f"H_C particle-hole symmetric ({name})", np.abs(U @ H.conj() @ U.conj().T + H).max(), 1e-12)


def solvers_agree():
    for name, p in CASES.items():
        worst_d, worst_f = 0.0, 0.0
        rgf = RGFFourTerminal(FourTerminalJunction(p), E)
        fps = FastPhaseSweep(FourTerminalJunction(p), E)
        for phi in (0.0, 1.3, np.pi):
            dense = {s: FourTerminalJunction(replace(p, phi=phi)).channels(E, side_name=s) for s in ('left', 'right')}
            for s in ('left', 'right'):
                r, f = rgf.channels_at_phi(phi, s), fps.channels_at_phi(phi, s)
                worst_d = max(worst_d, max(np.abs(np.asarray(dense[s][k]) - np.asarray(r[k])).max() for k in r))
                worst_f = max(worst_f, max(np.abs(np.asarray(f[k]) - np.asarray(r[k])).max() for k in r))
        _report(f"RGF == dense channels ({name})", worst_d, 1e-8)
        _report(f"RGF == FastPhaseSweep ({name})", worst_f, 1e-8)


def particle_hole_of_transmissions():
    Ep = np.array([3e-4, 2.7e-3, 0.03, 0.2])
    for name, p in CASES.items():
        r = RGFFourTerminal(FourTerminalJunction(p), np.concatenate([Ep, -Ep]))
        worst = 0.0
        for phi in (0.0, 1.1, np.pi):
            c, n = r.channels_at_phi(phi, 'right'), len(Ep)
            worst = max(worst, np.abs(c['ee'][n:] - c['hh'][:n]).max(), np.abs(c['he_cross'][n:] - c['eh_cross'][:n]).max())
        _report(f"T_ee(-E) = T_hh(E), T_he(-E) = T_eh(E) ({name})", worst, 1e-9)


def disorder_reaches_everything():
    """
    Disorder must reach BOTH solvers and the central fingerprint.  Both silently failed:
    LocalGreen used the clean slice, so the LDOS was disorder-blind; and central_fingerprint
    omitted the realisation, so scan()/ThermalPoint reused one seed's solve for every seed.
    """
    p = replace(CASES['rashba'], nx=12, disorder_W=0.5)
    seeds = (0, 1, 2)
    fps = {central_fingerprint(FourTerminalJunction(replace(p, disorder_seed=s)), np.zeros(1))
           for s in seeds}
    _report("central_fingerprint distinguishes disorder seeds", 0.0 if len(fps) == len(seeds) else 1.0, 0.5)
    A = [ldos(replace(p, disorder_seed=s), [0.0], np.pi, cols=[0, 5]).sum() for s in seeds]
    _report("LDOS responds to the disorder seed", 0.0 if max(A) - min(A) > 1e-9 else 1.0, 0.5)
    ch = [RGFFourTerminal(FourTerminalJunction(replace(p, disorder_seed=s)),
                          np.zeros(1)).channels_at_phi(np.pi, 'right')['ee'][0] for s in seeds]
    _report("channels respond to the disorder seed",
            0.0 if max(ch) - min(ch) > 1e-9 else 1.0, 0.5)


def phs_at_zero_energy():
    """T_ee = T_hh and T_eh = T_he at E = 0 exactly.  particle_hole_of_transmissions covers
    E != 0 on a symmetric grid; this covers E = 0 itself, which is where small eta breaks down."""
    for name, p in CASES.items():
        r = RGFFourTerminal(FourTerminalJunction(p), np.zeros(1))
        worst = max(phs_residual(r.channels_at_phi(phi, 'right'), np.zeros(1))
                    for phi in (0.0, 1.1, np.pi))
        _report(f"T_ee = T_hh and T_eh = T_he at E = 0 ({name})", worst, 1e-9)


def clean_wire():
    # SC ribbons fully decoupled: tc = 0 AND alpha = 0 (a Rashba bond keeps its SOC part even at t = 0)
    p = Params(model='rashba', nx=8, ny=5, t_n=1.0, mu_n=0.8, t_c=1.0, mu_c=0.8, t_s=1.0, mu_s=0.5, delta=0.3,
               tc_top=0.0, tc_bot=0.0, tc_barr=1.0, alpha=0.0, Bz=0.2, eta=1e-9)
    # energies chosen away from the subband thresholds (-0.73, -0.33, 0.0, 0.4 for these parameters)
    c = RGFFourTerminal(FourTerminalJunction(p), np.array([-0.1, 0.13, 0.25])).channels_at_phi(0.0, 'right')
    T = np.asarray(c['ee'])
    _report("clean wire: T_ee integer", np.abs(T - np.round(T)).max(), 1e-6)
    _report("clean wire: no Andreev (T_he + R_eh)", np.abs(c['he_cross']).max() + np.abs(c['eh_local']).max(), 1e-6)


def lead_solver():
    p = Params(model='rashba', nx=80, ny=12, t_n=1.0, mu_n=0.5, t_c=1.0, mu_c=0.05, t_s=1.0, mu_s=0.30, delta=0.5,
               tc_top=1.55, tc_bot=1.55, tc_barr=1.0, alpha=1.6, Bz_s=1.45, eta=1e-6)
    top = FourTerminalJunction(p).ribbon_top
    H0, V = top.H_onsite, top.V_hop
    M = H0.shape[0]
    z = (3e-3 + 1j * p.eta) * np.eye(M)[None]
    ep, em = leads.sancho_rubio(H0, V, z, p.max_iter, p.tol)
    d = max(np.abs(leads.eigenmode_eps(H0, V, z[0, 0, 0]) - ep[0]).max(),
            np.abs(leads.eigenmode_eps(H0, V.conj().T, z[0, 0, 0]) - em[0]).max())
    _report("eigenmode fallback == Sancho-Rubio (healthy energy)", d, 1e-9)
    z0 = (0.0 + 1j * p.eta) * np.eye(M)[None]
    cache, leads.CACHE_ENABLED = leads.CACHE_ENABLED, False
    e_plus, e_minus = leads.surface_eps_pair(H0, V, z0, p)
    leads.CACHE_ENABLED = cache
    _report("known E = 0 failure repaired (lead residual)",
            max(leads.residual(H0, V, z0, e_plus)[0], leads.residual(H0, V.conj().T, z0, e_minus)[0]), leads.ACCEPT_TOL)


def run_all():
    _RESULTS.clear()
    for f in (hermiticity_and_phs_of_H, solvers_agree, particle_hole_of_transmissions,
              phs_at_zero_energy, disorder_reaches_everything, clean_wire, lead_solver):
        f()
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    return all(_RESULTS)
