"""Tests of the array-returning interface (compute.py), scan/ThermalPoint and the finite-bias wrappers.
Run from the 4_terminal folder:  python tests/test_api.py      (about a minute)"""
import os, sys, tempfile, time
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import my_functions as myf

P = myf.Params(model='rashba', nx=24, ny=8, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
               delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
R = []


def check(name, ok, detail):
    R.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    E = np.linspace(-0.02, 0.02, 9)
    r = myf.RGFFourTerminal(myf.FourTerminalJunction(P), E)
    T1, T2 = myf.transmissions(P, E, 1.1), myf.transmissions(P, E, [0.0, 1.1])
    d = max(np.abs(T1[k] - r.channels_at_phi(1.1, 'right')[k]).max() for k in T1)
    check("transmissions == RGF", d == 0 and T2['ee'].shape == (2, len(E)), f"diff {d:.1e}")
    k0, _ = myf.linear_response(P, np.pi, 0.0); c0 = r.channels_at_phi(np.pi, 'right')
    k, g, ek, eg = myf.linear_response(P, np.pi, 1e-3, error=True)
    ks, _ = myf.linear_response(P, [np.pi, 0.0], 1e-3)
    check("linear_response", abs(k0 - (c0['ee'][4] + c0['he_cross'][4])) < 1e-9 and ks[0] == k and np.isfinite(ek),
          f"T->0 {k0:.4f}; kT=1e-3: {k:.4f} +- {ek:.1e}")
    A = myf.ldos(P, [0.0, 0.01], np.pi, cols=[0, 12])
    check("ldos", A.shape == (2, 2, P.ny) and (A > -1e-12).all(), f"shape {A.shape}")
    Eb, V = np.linspace(-0.5, 0.5, 201), np.linspace(-0.2, 0.2, 7)
    ch = myf.transmissions(P, Eb, 1.1, 'left')
    I1, I2 = myf.current(ch, Eb, V, -V, 2e-3), myf.currents(P, Eb, V, -V, 2e-3, 1.1)
    G1, G2 = myf.differential_conductance(ch, Eb, V, -V, 2e-3), myf.partial_G_vectorized(ch, Eb, V, 1e-5, 2e-3, 'anti')
    check("finite bias", np.abs(I1 - I2).max() < 1e-12 and max(np.abs(a - b).max() for a, b in zip(G1, G2)) == 0, "core == wrappers")
    f = myf.ThermalPoint(P, kT=0.0, phis=(np.pi, 0.0))
    grid = {'mu_s': [0.1, 0.2], 'Bz_s': [0.8, 0.94, 1.0]}          # ribbon parameter last
    with tempfile.TemporaryDirectory() as tmp:
        s1 = myf.scan(f, grid)
        s2 = myf.scan(f, grid, n_workers=2, save=os.path.join(tmp, 's.npz'))
        t1 = time.perf_counter(); s3 = myf.scan(f, grid, n_workers=2, save=os.path.join(tmp, 's.npz')); tr = time.perf_counter() - t1
    direct = myf.linear_response(replace(P, mu_s=0.2, Bz_s=1.0), [np.pi, 0.0], 0.0)
    # serial and parallel runs may differ at round-off level (different BLAS threading / reuse order)
    d_par = np.abs(s1['values'] - s2['values']).max()
    d_res = np.abs(s2['values'] - s3['values']).max()
    d_dir = np.abs(s1['values'][1, 2, [0, 2]] - np.asarray(direct[0])).max()
    check("scan + ThermalPoint", s1['values'].shape == (2, 3, 5) and d_par < 1e-10 and d_res == 0 and d_dir < 1e-10 and tr < 1.0,
          f"shape {s1['values'].shape}, |parallel - serial| {d_par:.1e}, |resumed - saved| {d_res:.1e}, "
          f"|scan - direct| {d_dir:.1e}, resume {tr:.2f} s")
    print(f"\n{sum(R)}/{len(R)} tests passed ({time.perf_counter() - t0:.0f} s)")
