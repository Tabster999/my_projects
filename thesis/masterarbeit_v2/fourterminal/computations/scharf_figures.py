"""
Reproduces three figures of Scharf et al. (PRB 99, 214503) with this package, at the paper's own
parameters (L = 2000 nm, W = 100 nm, Delta = 0.25 meV, lambda_soc = 16 meV nm, m = 0.038 m0,
a = 20 nm, [100] phase bias, in-plane field at the optimal angle).

    fig7()      real-space LDOS for three Rashba/Dresselhaus combinations     -> Fig. 7
    fig8abcd()  LDOS at the junction end vs (E, phi) for four Zeeman fields    -> Figs. 8(a)-(d)
    fig8ef()    LDOS(E=0) at the end and in the middle vs (phi, E_Z), plus     -> Figs. 8(e),(f)
                the zero-bias dI/dV  (the LDOS itself, not its 2nd derivative)

All heavy results are cached in .npz files, so re-running only redraws.  fig8ef is the expensive
one (a few minutes); the others take tens of seconds.  Run as a script for all three.
"""
# %% --- IMPORTS AND COMMON SETTINGS ---
import os

import numpy as np
import matplotlib.pyplot as plt

import scharf_ldos as S          # same folder; brings in my_functions and PhysParams

LAM = 16.0                       # total SOC strength, meV nm
OUT = "."                        # where figures and .npz files go


def base(alpha=14.3, beta=7.3, **kw):
    """The paper's junction; alpha/beta default to its theta_soc = 0.15 pi case."""
    return S.PhysParams(L=2000.0, W=100.0, tc_barr=0.02, soc_axis='100',
                        alpha=alpha, beta=beta, **kw)


# %% --- Fig. 7: real-space Majorana bound states for three SOC combinations ---
def fig7(B=1.73, fname="scharf_fig7.png"):
    cases = [(0.00, r"(a) pure Rashba,  $\theta_{soc}=0$"),
             (0.15, r"(b) mixed,  $\theta_{soc}=0.15\pi$"),
             (0.25, r"(c) $\alpha=\beta$,  $\theta_{soc}=0.25\pi$")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, (ts, title) in zip(axs, cases):
        al, be = LAM * np.cos(ts * np.pi), LAM * np.sin(ts * np.pi)
        pp = base(alpha=al, beta=be, B=B)
        x, y, A = S.real_space(np.pi, B, pp=pp, every=1, show=False)
        im = ax.pcolormesh(x, y, A, shading='nearest', cmap='viridis')
        fig.colorbar(im, ax=ax)
        for xi in (-pp.W / 2, pp.W / 2):
            ax.axvline(xi, color='w', ls='--', lw=1)
        ax.set_xlim(-pp.W / 2, pp.W / 2)
        ax.set_ylim(-pp.L / 2, pp.L / 2)
        ax.set_xlabel("x [nm]  (across)")
        ax.set_ylabel("y [nm]  (along)")
        prof = A.sum(axis=1)
        ax.set_title(f"{title}\n$\\alpha$={al:.1f}, $\\beta$={be:.1f} meV nm;  "
                     f"end/mid = {prof[0] / prof[len(prof) // 2]:.0f}", fontsize=10)
    pp = base()
    fig.suptitle(r"Scharf Fig. 7: $|\psi|^2$ at $E=0$, $\phi=\pi$, "
                 f"$E_Z$ = {pp.g * 5.78e-2 * B / 2:.2f} meV, L = {pp.L:.0f} nm, W = {pp.W:.0f} nm",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.show()


# %% --- Figs. 8(a)-(d): LDOS at the end vs energy and phase ---
def fig8abcd(Bs=(0.0, 0.8, 1.6, 2.4), n_E=41, n_phi=25, fname="scharf_fig8abcd.png",
             cache="scharf_fig8abcd.npz"):
    pp = base()
    E = np.linspace(-0.3, 0.3, n_E)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    path = os.path.join(OUT, cache)
    data = dict(np.load(path)) if os.path.exists(path) else {}
    for B in Bs:
        key = f"end_B{B}"
        if key not in data:
            _, _, end, _ = S.spectrum(B, pp=pp, E_meV=E, phis=phis, show=False)
            data[key] = end
            np.savez(path, E=E, phis=phis, Bs=np.array(Bs), **data)
            print(f"B = {B} T done", flush=True)
    fig, axs = plt.subplots(1, len(Bs), figsize=(4.3 * len(Bs), 4), sharey=True)
    for ax, B in zip(np.atleast_1d(axs), Bs):
        im = ax.pcolormesh(E / pp.Delta, phis / np.pi, data[f"end_B{B}"],
                           shading='nearest', cmap='inferno')
        fig.colorbar(im, ax=ax)
        EZ = pp.g * 5.78e-2 * B / 2
        ax.axvline(0, color='w', ls=':', lw=0.8)
        ax.set_xlabel(r"$E/\Delta$")
        ax.set_title(f"$E_Z$ = {EZ:.2f} meV  ({EZ / pp.E_T:.2f} $E_T$)", fontsize=10)
    np.atleast_1d(axs)[0].set_ylabel(r"$\phi/\pi$")
    fig.suptitle("Scharf Fig. 8(a)-(d): LDOS at the junction END vs energy and phase "
                 f"(L={pp.L:.0f} nm, W={pp.W:.0f} nm, alpha={pp.alpha}, beta={pp.beta} meV nm)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.show()


# %% --- Figs. 8(e),(f): LDOS(E=0) at the end and in the middle vs (phi, E_Z) ---
def fig8ef(n_phi=41, b_max=5.0, n_B=26, fname="scharf_fig8ef.png", cache="scharf_fig8_map.npz"):
    pp = base()
    R = S.run_map(pp=pp, phis=np.linspace(0, 2 * np.pi, n_phi),
                  b_vals=np.linspace(0.0, b_max, n_B), save=os.path.join(OUT, cache))
    phis, EZ = R['phis'], R['E_Z']
    panels = [('ldos_end', "(e) LDOS($E$=0) at the junction END", 'inferno'),
              ('ldos_mid', "(f) LDOS($E$=0) in the MIDDLE", 'inferno'),
              ('dIdV', r"zero-bias $dI/dV$  [$e^2/h$]", 'magma')]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, title, cmap) in zip(axs, panels):
        Z = R[key]
        im = ax.pcolormesh(phis / np.pi, EZ, Z, shading='nearest', cmap=cmap,
                           vmax=np.nanpercentile(Z, 99))
        fig.colorbar(im, ax=ax)
        ax.axhline(pp.E_T, color='w', ls='--', lw=1.2)
        ax.set_xlabel(r"$\phi/\pi$")
        ax.set_ylabel(r"$E_Z$  [meV]")
        ax.set_title(title, fontsize=11)
    axs[0].text(0.05, pp.E_T * 1.05, r"$E_T$", color='w', fontsize=9)
    fig.suptitle(f"Scharf Fig. 8(e),(f): L={pp.L:.0f} nm, W={pp.W:.0f} nm, alpha={pp.alpha}, "
                 f"beta={pp.beta} meV nm, Delta={pp.Delta} meV, eta={pp.eta_rel} Delta", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.show()
    # contrast along phi = pi, the quantity that tells you where the bound states live
    j = int(np.argmin(np.abs(phis - np.pi)))
    ratio = R['ldos_end'] / R['ldos_mid']
    print("end/middle contrast at phi = pi:")
    for i in range(0, len(EZ), max(1, len(EZ) // 8)):
        print(f"   E_Z = {EZ[i]:.2f} meV ({EZ[i] / pp.E_T:.2f} E_T): {ratio[i, j]:6.1f}")


# %% --- RUN ---
if __name__ == "__main__":
    base().summary()
    fig7()
    fig8abcd()
    fig8ef()
