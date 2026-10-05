# Context for Claude Code — thesis/fourterminal

Master's thesis code: NEGF transport through a four-terminal Josephson junction; goal is the half-integer
thermal-conductance plateau (κ/κ₀ = 0.5 with G/G₀ = 0 at φ = π, Gresta et al., PRB 114, 125405) in a
Rashba model. See README.md for structure and usage.

## Rules
* Keep existing public names (Params fields, RGFFourTerminal.channels_at_phi, …); scripts depend on them.
* After changing hamiltonians/leads/rgf/transport run `run_all()` and `tests/test_api.py`; all must pass.
* Never let a failed lead solve pass silently (NaN is intended). Report `myf.leads.N_UNRELIABLE` in scans.
* Computation returns arrays; plotting stays in the scripts.

## Physics established so far (Rashba model)
* Working point: nx=80, ny=12, Δ=0.35, α=1.2, tc=1.1, μ_c=−0.03, μ_n=1, μ_s≈0.15, B_z,s≈0.94, η=1e-5 →
  κ=0.497 at k_BT=1e-3 with κ(φ=0)=0.003; connected T→0 plateau in (B_z,s, μ_s).
* Plateau width: Majorana decay length ξ vs nx (`coherence_length`; nx ≳ 12ξ at T→0, ≳ 7ξ at k_BT=1e-3).
  Larger Δ and α shorten ξ.
* Plateau height: interface matching (tc ≈ 1.1–1.2, μ_n ≈ 1).
* Phase contrast: the centre is depleted (no Dirac point), so the φ=0 gap decays exponentially with ny;
  contrast is lost for ny > 12 (κ(0): 0.006 at ny=12, 0.45 at ny=20). No SOC/Zeeman/mass variant fixes this.
* Dirac model reproduces the paper's Fig. 8 with μ_n ≈ 1 (unstated in the paper).

## Scharf model (`computations/scharf_ldos.py`; Scharf et al., PRB 99, 214503)
* LDOS and transport need DIFFERENT settings and cannot share one parameter set. The LDOS uses
  Scharf's tc_barr=0 (decoupled tunnelling probes) and η=0.05Δ, and both make κ and G vanish
  identically. κ/G use `tc_barr_transport=0.2`, `eta_rel_transport=1e-5` through
  `PhysParams.params_transport()`; the coupling must stay weak (tc_barr≈1 → κ=0.001, the end states
  hybridise with the leads).
* L=2000 nm, W=100 nm, α=14.3, β=7.3 meV nm, E_Z=0.5 meV, φ=π → κ/κ₀=0.491 with G/G₀=0.0001
  (T_ee=T_he=0.245 to four digits: the Majorana condition), end/mid LDOS = 7.9. At φ=0 all channels
  are <1e-4 and the LDOS ratio is 2.4.
* It is a RESONANCE, not a plateau: its width is the hybridisation splitting of the two end
  Majoranas across L, not a topological gap, because Scharf's ribbons are trivial (Bz_s=0) and the
  Majoranas are localised at y=±L/2. κ falls to 0.44 at kT=0.1 μeV (≈1 mK) and 0.02 at 1 μeV.
  Contrast with the Gresta working point above, where a propagating chiral Majorana keeps κ=0.497
  at k_BT=1e-3.
