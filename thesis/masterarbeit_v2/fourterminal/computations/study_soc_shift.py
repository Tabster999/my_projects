"""
STUDY: what changes if the constant SOC shift E_so = m(alpha^2 + beta^2)/2 hbar^2 is left out of the
onsite energy (my_functions.hamiltonians.onsite_block adds it, so mu is measured from the bottom of the
lower SOC band).  Every region has the same alpha, so dropping the shift == raising mu_S, mu_N and mu_C
by E_so.  Both conventions on the SAME field grid (a = 20 nm, L = 2 um, tc_barr = 0.2, phi = pi):
the gaps along phi = pi and kappa / G at E = 0.
"""
# %% --- IMPORTS AND PARAMETERS ---
import numpy as np
import matplotlib.pyplot as plt
import tools as st

HB2M = 76.1996 / 0.038                                   # hbar^2 / m, meV nm^2
CASES = {'W100 a16': dict(W=100.0, alpha=16.0), 'W100 a32': dict(W=100.0, alpha=32.0),
         'W300 a16': dict(W=300.0, alpha=16.0)}


def no_shift(q):
    """The same junction without the constant SOC shift: every mu raised by E_so."""
    E_so = (q.alpha ** 2 + q.beta ** 2) / (2 * HB2M)
    return st.replace(q, mu_S=q.mu_S + E_so, mu_N=q.mu_N + E_so, mu_C=q.mu_centre + E_so)


# %% --- RUN ---
if __name__ == "__main__":
    RES = {}
    for name, c in CASES.items():
        q0 = st.case(c['W'], alpha=c['alpha'])
        EZ_line, EZ_tr = st.field_grid(q0, 41), st.field_grid(q0, 121)     # same grids for both conventions
        tag = name.replace(' ', '_')
        E_so = c['alpha'] ** 2 / (2 * HB2M)
        print(f"\n{name}: E_so = {E_so * 1e3:.0f} ueV;  v_F(junction) with shift {np.sqrt(2 * q0.mu_centre * HB2M):.1f},"
              f" without {np.sqrt(2 * (q0.mu_centre + E_so) * HB2M):.1f} meV nm")
        for conv, q in (('with shift', q0), ('no shift', no_shift(q0))):
            line = st.phase_line(q, EZ_line, cache=f'scharf_absS_line_{tag}_{conv.replace(" ", "")}')
            t = st.transport_line(q, EZ_tr, cache=f'scharf_absS_tr_{tag}_{conv.replace(" ", "")}')
            RES[(name, conv)] = (line, t)
            topo = line['Q'] < 0
            i = st.best_topological(line)
            pk = st.kappa_peaks(t)
            s = st.peak_stats(t)
            print(f"  {conv:10s}: ABS {st.count_abs(q)}, Q = -1 for {line['E_Z'][topo].min():.3f}..{line['E_Z'][topo].max():.3f} meV,"
                  f" Delta_top {line['gap'][i] * 1e3:.1f} ueV at {line['E_Z'][i]:.3f};  Majorana-like {s['good']}/{s['n']}")
            print(f"              kappa peaks at {np.round(t['E_Z'][pk], 3)}")
            print(f"              heights {np.round(t['kappa'][pk], 3)},  |G|/kappa {np.round(np.abs(t['G'][pk]) / t['kappa'][pk], 3)}")


# %% --- FIGURE ---
if __name__ == "__main__":
    fig, axs = plt.subplots(len(CASES), 2, figsize=(14, 3 * len(CASES)))
    for row, name in zip(axs, CASES):
        for conv, ls in (('with shift', '-'), ('no shift', '--')):
            line, t = RES[(name, conv)]
            row[0].plot(line['E_Z'], line['gap'] * 1e3, ls, label=f'Delta_top, {conv}')
            row[1].plot(t['E_Z'], t['kappa'], ls, lw=1, label=rf'$\kappa$, {conv}')
        row[0].set_ylabel(f'{name}\ngap [$\\mu$eV]'); row[1].set_ylabel(r'$\kappa/\kappa_0$')
        row[0].legend(fontsize=7); row[1].legend(fontsize=7); row[1].set_ylim(0, 1.05)
    axs[-1, 0].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$)'); axs[-1, 1].set_xlabel(r'$E_Z$ [meV]  ($\phi = \pi$, E = 0)')
    plt.tight_layout(); st.save_fig(fig, 'soc_shift'); plt.show()
