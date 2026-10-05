# 4_terminal — NEGF transport through a four-terminal Josephson junction

Central `nx × ny` normal region, superconducting ribbons top/bottom (phase difference φ), normal leads
left/right. BdG basis (c↑, c↓, c↓†, −c↑†). Two models share all geometry/lead/solver code:
`model='rashba'` (2DEG, Rashba/Dresselhaus SOC, Zeeman, s-wave pairing; `Bz_s` = Zeeman in the ribbons)
and `model='dirac'` (Gresta et al., PRB 114, 125405, Eq. 1).

## Quick start
```python
import numpy as np, my_functions as myf
p = myf.Params(model="rashba", nx=80, ny=12, mu_n=1.0, mu_c=-0.03, mu_s=0.15, delta=0.35,
               tc_top=1.1, tc_bot=1.1, tc_barr=1.0, alpha=1.2, Bz_s=0.94, eta=1e-5)
kappa, G = myf.linear_response(p, np.pi, kT=1e-3)            # thermal / electrical linear response
T = myf.transmissions(p, np.linspace(-0.01, 0.01, 41))      # channel transmissions (dict of arrays)
res = myf.scan(myf.ThermalPoint(p, kT=0.0, phis=(np.pi, 0.0)),
               {"mu_s": [0.1, 0.2], "Bz_s": np.linspace(0.6, 1.4, 17)}, n_workers=8, save="scan.npz")
```
Everything returns arrays; plotting is left to you. Examples: `computations/01…06_*.py`.

## Structure (`my_functions/`)
| module | contents |
|---|---|
| `params.py` | `Params` (all settings; optional per-region `alpha_c/alpha_n/alpha_s`, `Bz_n`) |
| `hamiltonians.py` | onsite/hopping blocks of both models, region handling |
| `leads.py` | lead surface solutions: Sancho-Rubio → Newton → eigenmode fallback, residual-checked, cached |
| `junction.py` | `FourTerminalJunction` (geometry; dense `channels()` as reference) |
| `rgf.py` | `RGFFourTerminal` — the solver (`channels_at_phi`, `green_blocks`, `with_ribbons`) |
| `fast_phase_sweep.py` | `FastPhaseSweep` — dense reference for small systems / checks |
| `transport.py` | finite bias: `current`, `differential_conductance` (+ old names as wrappers); thermal: `linear_response_nodes_phs`, `thermal_from_channels_phs`, `thermal_error_phs` |
| `compute.py` | one-call functions: `transmissions`, `linear_response`, `ldos`, `currents`, `conductances` |
| `scans.py` | `scan` (parallel, resumable, any quantity), `ThermalPoint`; low level `thermal_point`, `thermal_scan`, `make_pool` |
| `spectra.py` | `chern_number`, `ribbon_gap`, `coherence_length`, `edge_state_profile`, `junction_spectrum_kx`, `LocalGreen` |
| `checks.py` | `run_all()` — physics and solver regression checks |

## Numerics you should know
* **Lead solver.** Sancho-Rubio loses accuracy at E ≈ 0 when the ribbons' edge gap is far below η (the
  single-Majorana regime). Every lead solution is checked against its Dyson equation; failures are
  Newton-refined, then replaced by the eigenmode solution, and set to **NaN** if still failing (never a
  silently wrong number). Count: `myf.leads.N_UNRELIABLE`; threshold `myf.leads.ACCEPT_TOL` (1e-8).
* **Thermal integral.** Trapezoid rule on a uniform grid in x = E/kT (step h = 0.5, |x| ≤ 16), negative
  energies from particle-hole symmetry (33 energies). Accurate to ≲2e-3 where T(E) is smooth; sharp
  resonances (φ = 0 above the minigap, open normal channels) may need h = 0.25. `linear_response(...,
  error=True)` / `thermal_error_phs` give a free error estimate. (The former Gauss-Legendre rule was off
  by up to 0.14 at φ = 0.)
* **Speed.** In `scan`, put ribbon-only parameters (mu_s, Bz_s, delta, t_s, m0) last: each worker then
  reuses its central solve. Sweeps of central parameters (mu_c, tc, …) reuse the cached leads.

## Checks and tests
```
python -c "from my_functions.checks import run_all; run_all()"    # 19 checks, ~1 min
python tests/test_api.py                                         # interface, scan, wrappers
```
