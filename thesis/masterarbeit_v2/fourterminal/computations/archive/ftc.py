import sys, os, time, numpy as np, warnings
sys.path.insert(0, '/tmp/repo/thesis/4_terminal')
from dataclasses import replace
import my_functions as myf
from my_functions import leads
warnings.simplefilter('ignore')
KT = 1e-3
PTS = {'A': (myf.Params(model='rashba', nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0, mu_s=0.15,
                        delta=0.35, phi=np.pi, tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5), 0.94),
        'B': (myf.Params(model='rashba', nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=0.05, t_s=1.0, mu_s=0.30,
                        delta=0.5, phi=np.pi, tc_top=1.55, tc_bot=1.55, tc_barr=1.0, alpha=1.6, Bz_s=1.55, eta=1e-5), 1.55)}
if __name__ == "__main__":
    f = '/tmp/ftc.npz'
    R = dict(np.load(f)) if os.path.exists(f) else {}
    t0 = time.perf_counter()
    for tag, (p0, B) in PTS.items():
        key = f'{tag}_centre'
        if key not in R:
            k, g, ek, eg = myf.linear_response(p0, [np.pi, 0.0], KT, error=True)
            R[key] = np.array([k[0], g[0], k[1], g[1], ek[0], ek[1]])
            np.savez(f, **R); print(f"set {tag} centre: kappa(pi)={k[0]:.4f}+-{ek[0]:.0e} G={g[0]:+.4f} | kappa(0)={k[1]:.4f} G(0)={g[1]:+.4f}", flush=True)
        if time.perf_counter() - t0 > float(sys.argv[1]): break
