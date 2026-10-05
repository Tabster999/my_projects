# Context for Claude Code — thesis/4_terminal

Master's thesis code: NEGF transport through a four-terminal Josephson junction; goal is the half-integer
thermal-conductance plateau (κ/κ₀ = 0.5 with G/G₀ = 0 at φ = π, Gresta et al., PRB 114, 125405) in a
Rashba model. See README.md for structure and usage. The older `thesis/masterarbeit_v2/fourterminal/` is legacy.

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
