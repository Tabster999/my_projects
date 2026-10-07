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
  Scharf's tc_barr=0 and η=0.05Δ, and both make κ and G vanish identically (the saved map had
  κ≡0). κ/G go through `PhysParams.params_transport()` with `tc_barr_transport≈0.2`,
  `eta_rel_transport=1e-5`.
* The probe coupling must stay WEAK. At tc_barr=1, κ collapses to 0.001 while the LOCAL Andreev
  conductance rises to the quantized 1.998≈2e²/h: strong coupling buys local spectroscopy, weak
  coupling buys nonlocal transport, never both. Nonlocal transport needs δ ≳ Γ, and the Majorana
  splitting δ ~ Δe^(−L/ξ) is exponentially small, so Γ must be matched down to it. This is the
  OPPOSITE of the Gresta case, where the carrier propagates and you want a transparent, mode-matched
  contact (tc≈1.1, μ_n≈1).
* κ OSCILLATES with E_Z and its MAXIMA are quantized at 1/2 — it is not a plateau. At L=2000 nm,
  W=100 nm, α=14.3, β=7.3, soc_axis='100', φ=π: 7 resonances with κ=0.474–0.496 over
  E_Z=0–0.8 meV, period ≈0.055 meV, each ≈0.03 meV wide and smooth. This is the Majorana
  oscillation δ ~ Δe^(−L/ξ)cos(k_F L).
* EXACT complementarity between local and nonlocal Majorana weight: at the κ nodes the local
  Andreev conductance is 1.98≈2e²/h; at the κ peaks it falls to ≈0.5. The weight trades between
  local reflection and nonlocal transmission.
* κ is FRAGILE in temperature, because the width is δ and not a gap: at E_Z=0.5, φ=π,
  κ = 0.491 → 0.44 at k_BT=0.1 μeV (≈1 mK) → 0.019 at 1 μeV.
* |G|/κ DISCRIMINATES genuine Majoranas from LDOS false positives, and is resonance-INdependent:
  T_ee=T_he holds to 4–5 digits everywhere the Majorana exists, not just on resonance. On the
  61×61 '100' map, of 1064 points with κ>0.05, 1011 are neutral (|G|/κ<0.1) and 10 are charged
  (>0.5). All 10 pass Scharf's own curv_end<0 criterion, yet have end/mid contrast 1.7–3.3
  (vs 7.7–13.0 for the genuine ones) and sit at E_Z ≳ 0.9 E_T — delocalized zero-energy states.
  Caveat: |κ−0.5| alone is a valid POSITIVE only; a genuine Majorana off resonance also gives κ≈0.
  Maps cached in `computations/scharf_kappa_map_{100,110}.npz` (61×61, ~16 min each).
* soc_axis matters: '110' puts (α−β) on the propagation axis, '100' puts √(α²+β²). With β=7.3
  that halves the effective SOC at '110' — κ>0.05 on 9.6% of the map vs 28.6% at '100'.

## Combining Gresta and Scharf (all tested, all negative)
* A κ PLATEAU needs a PROPAGATING Majorana, hence Chern≠0 in the ribbons, hence OUT-OF-PLANE
  Zeeman. An in-plane field in the ribbons makes the 2D spectrum NODAL at any magnitude (bulk gap
  ~1e-4 for Bxy_s=0.94–3.0) and destroys an existing Chern gap even alongside Bz_s=0.94.
  `chern_number` still returns −1 on a nodal spectrum, where it is MEANINGLESS — always check the
  bulk gap first in any scan.
* In-plane LEAKAGE into the SC is tolerable: at Bz_s=0.94 the gap survives to Bxy_s≈0.3
  (0.267→0.039) and dies by 0.5. A mostly-out-of-plane field with a modest tilt is fine, which
  matters because a real magnet fields the whole sample.
* Bz_s (ribbons) + Bxy_c (centre, Scharf's field) do NOT combine. Bxy_c=0.1 already takes κ(π)
  0.474→0.148 and κ(0) 0.003→0.652 (contrast inverts); no rescue at ny=20. The two mechanisms
  need OPPOSITE central regions — Gresta depleted (μ_c=−0.03), Scharf populated (μ_N=0.7 meV) — and
  any Bxy_c large enough for Scharf physics (≫|μ_c|) fills the centre with ordinary channels.
  Watch for κ≈G there (e.g. κ=0.447, G=0.411 at Bxy_c=0.8): an ordinary channel passing near 0.5.
* Dresselhaus hurts the Gresta plateau in BOTH orientations, for different reasons. '110' closes
  the Chern gap through (α−β) and ξ diverges (5.5→49.7 sites for β=0→1.1). '100' keeps ξ short
  (5.5→4.5, since SOC adds in quadrature) but κ(π) overshoots to 0.91–0.99 with G(π)≈0.33, i.e.
  it stops being charge-neutral. Larger nx does NOT help: κ(π) 0.358→0.192 for nx=80→240 at β=0.8.
