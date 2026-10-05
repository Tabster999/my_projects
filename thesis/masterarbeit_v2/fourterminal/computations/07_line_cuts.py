# %% --- IMPORTS ---
import numpy as np
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

try:
    import my_functions as myf
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    sys.path.insert(0, str(next(p for p in (_here, *_here.parents) if (p / "my_functions").is_dir())))
    import my_functions as myf

# %% --- PARAMETERS ---
N_WORKERS = 1          # set to your core count
OUT = "results"        # output folder for the .npz files

SETS = {
    # A: realistic working point (Delta = 0.35)
    "A": dict(
        p=myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=-0.03, t_s=1.0,
                     mu_s=0.15, delta=0.35, phi=np.pi, tc_top=1.0, tc_bot=1.0, tc_barr=1.0,
                     alpha=1.2, Bz_s=0.94, eta=1e-5, lead_refine=True, kT=1e-3),
        Bs=np.round(np.linspace(0.00, 2.00, 61), 4),
        mus=np.round(np.linspace(0.0, 2.0, 61), 4),
        mcs=np.round(np.linspace(-0.12, 0.10, 45), 4),
        mu_cuts=(0.15, 0.40),
    ),
    # B: wide-plateau point (Delta = 0.5)
    "B": dict(
        p=myf.Params(model="rashba", nx=80, ny=12, t_n=1.0, mu_n=1.0, t_c=1.0, mu_c=0.05, t_s=1.0,
                     mu_s=0.30, delta=0.5, phi=np.pi, tc_top=1.0, tc_bot=1.0, tc_barr=1.0,
                     alpha=1.6, Bz_s=1.55, eta=1e-5, lead_refine=True, kT=1e-3),
        Bs=np.round(np.linspace(0.00, 2.00, 61), 4),
        mus=np.round(np.linspace(0.00, 2.00, 61), 4),
        mcs=np.round(np.linspace(-0.10, 0.20, 45), 4),
        mu_cuts=(0.30, 0.60),
    ),
}
NYS = np.arange(6, 25, 2)

# %% --- COMPUTE ---
if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    for tag, C in SETS.items():
        p0 = C["p"]
        f = myf.ThermalPoint(p0, kT=0.0, phis=(np.pi, 0.0))
        jobs = {
            "map":  {"mu_s": C["mus"], "Bz_s": C["Bs"]},          # ribbon parameter last
            "muc":  {"mu_c": C["mcs"]},
            "cut1": {"mu_s": [C["mu_cuts"][0]], "Bz_s": C["Bs"]},
            "cut2": {"mu_s": [C["mu_cuts"][1]], "Bz_s": C["Bs"]},
            "ny":   {"ny": NYS},
        }
        for name, grid in jobs.items():
            res = myf.scan(f, grid, n_workers=N_WORKERS, save=f"{OUT}/sweep_{tag}_{name}.npz")
            bad = int(np.nansum(res["values"][..., -1]))
            print(f"set {tag}, {name}: {res['done'].sum()}/{res['done'].size} points, failed lead energies: {bad}")

    for tag, C in SETS.items():
        p0, Bs, mus = C["p"], C["Bs"][::2], C["mus"][::2]
        gap = np.array([[myf.ribbon_gap(replace(p0, Bz_s=float(b), mu_s=float(m)), nk=31) for b in Bs] for m in mus])
        xi = np.array([[myf.coherence_length(replace(p0, Bz_s=float(b), mu_s=float(m))) for b in Bs] for m in mus])
        np.savez(f"{OUT}/diag_{tag}.npz", gap=gap, xi=xi, gB=Bs, gm=mus)
        print(f"set {tag}: diagnostics done")
"""Plot the sweeps: one Fig. 8-style figure per parameter set, plus a comparison of the two maps."""
# %% --- LOAD + PLOT ---
def load(tag, name):
    d = np.load(f"{OUT}/sweep_{tag}_{name}.npz", allow_pickle=False)
    return d["values"], {k[5:]: d[k] for k in d.files if k.startswith("axis_")}

for tag, C in SETS.items():
    p0 = C["p"]
    V, ax_map = load(tag, "map")                       # (n_mu, n_B, 5)
    kappa, G, kappa0 = V[..., 0], V[..., 1], V[..., 2]
    mu, Bz = ax_map["mu_s"], ax_map["Bz_s"]
    strict = (np.abs(kappa - 0.5) < 0.05) & (np.abs(G) < 0.05) & (kappa0 < 0.05)
    print(f"set {tag}: {strict.sum()}/{strict.size} points strictly half-quantized "
          f"({100*strict.mean():.0f}%), failed lead energies: {int(np.nansum(V[..., -1]))}")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    a = fig.add_subplot(gs[0, 0])                       # (a) map
    im = a.pcolormesh(Bz, mu, kappa, shading="nearest", cmap="inferno", vmin=0, vmax=1.2)
    fig.colorbar(im, ax=a)
    yy, xx = np.where(strict)
    a.plot(Bz[xx], mu[yy], "o", ms=5, mfc="none", mec="lime", mew=1.2,
           label=f"half-quantized ({strict.sum()}/{strict.size})")
    try:
        d = np.load(f"{OUT}/diag_{tag}.npz")
        a.contour(d["gB"], d["gm"], np.log10(d["gap"]), levels=[np.log10(p0.eta)], colors="cyan", linewidths=1.2)
    except FileNotFoundError:
        pass
    mm = np.linspace(mu.min(), mu.max(), 200)
    a.plot(np.hypot(p0.delta, mm - p0.alpha**2 / 4), mm, "w--", lw=1)     # topological transition
    a.set_xlim(Bz.min(), Bz.max()); a.set_ylim(mu.min(), mu.max())
    a.set_xlabel(r"$B_{z,s}$"); a.set_ylabel(r"$\mu_s$"); a.legend(fontsize=7, loc="upper left")
    a.set_title(r"(a) $\kappa/\kappa_0$ at $\phi=\pi$", fontsize=10)

    for axi, (name, xlab, ttl) in zip(
            (fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])),
            (("muc", r"$\mu_c$", "(b) $\\mu_c$ sweep"),
             ("cut1", r"$B_{z,s}$", f"(c) $\\mu_s$ = {C['mu_cuts'][0]}"),
             ("cut2", r"$B_{z,s}$", f"(d) $\\mu_s$ = {C['mu_cuts'][1]}"))):
        W, axes = load(tag, name)
        x = axes[xlab.strip("$\\").replace("_{z,s}", "z_s").replace("mu_c", "mu_c")] if False else axes[list(axes)[-1]]
        W = W.reshape(-1, W.shape[-1])
        axi.plot(x, W[:, 0], "b-", label=r"$\kappa/\kappa_0(\pi)$")
        axi.plot(x, W[:, 2], ":", color="tab:cyan", lw=2, label=r"$\kappa/\kappa_0(0)$")
        axi.plot(x, W[:, 1], "r-", label=r"$G/G_0(\pi)$")
        axi.axhline(0.5, color="gray", ls=":", lw=0.8); axi.axhline(0, color="gray", lw=0.5)
        axi.set_xlabel(xlab); axi.set_title(ttl, fontsize=10); axi.set_ylim(-0.1, 1.3)
    fig.axes[1].legend(fontsize=7)

    e = fig.add_subplot(gs[1, 1])                       # (e) ny
    W, axes = load(tag, "ny")
    e.plot(axes["ny"], W[:, 0], "o-", color="tab:blue", label=r"$\kappa/\kappa_0(\pi)$")
    e.plot(axes["ny"], W[:, 2], "o:", color="tab:cyan", lw=2, label=r"$\kappa/\kappa_0(0)$")
    e.axhline(0.5, color="gray", ls=":", lw=0.8)
    e.set_xlabel(r"$n_y$"); e.set_ylim(-0.05, 1.0); e.legend(fontsize=7)
    e.set_title("(e) junction width", fontsize=10)

    g = fig.add_subplot(gs[1, 2])                       # G map
    im = g.pcolormesh(Bz, mu, G, shading="nearest", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    fig.colorbar(im, ax=g); g.set_xlabel(r"$B_{z,s}$"); g.set_ylabel(r"$\mu_s$")
    g.set_title(r"$G/G_0$ at $\phi=\pi$", fontsize=10)

    fig.suptitle(myf.param_title(p0, extra=f"set {tag}: T -> 0 sweeps"), fontsize=8, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/sweeps_{tag}.png", dpi=130)
    plt.show()
# %%
