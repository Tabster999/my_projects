# Context for Claude Code — thesis/fourterminal

Master's thesis code: NEGF transport through a four-terminal Josephson junction.
**TWO DIFFERENT SYSTEMS share this package.  Their numbers are NOT comparable — check which
one a statement belongs to before reusing it.**  Current focus: the SCHARF model.
See README.md for structure and usage.

| | Gresta / Domínguez | Scharf |
|---|---|---|
| paper | PRB 114, 125405 | PRB 99, 214503 |
| driven from | `Params` directly | `PhysParams` in `computations/scharf_ldos.py` |
| units | dimensionless, t = 1 | physical meV/nm, t = 2.488 meV |
| geometry | nx=80 × ny=12 | nx=100 × ny=5  (L=2000 nm × W=100 nm) |
| Zeeman | `Bz_s`=0.94, OUT-of-plane, in the RIBBONS | `Bxy_c` in-plane, in the CENTRE; `Bz_s`=0 |
| ribbons | TOPOLOGICAL, Chern = −1 | TRIVIAL, Chern = 0 |
| Majoranas | PROPAGATING chiral mode along the junction | LOCALISED pair at the ends y = ±L/2 |
| α | 1.2 (units of t·a) | 14.3 meV·nm = 0.2865 t·a |
| centre | DEPLETED, μ_c = −0.03 | POPULATED, μ_N = 0.7 meV |
| κ = 1/2 is | a PLATEAU, protected by a bulk gap | a RESONANCE, width = the MBS splitting δ |
| probe contact | transparent: tc ≈ 1.1, μ_n ≈ 1 | WEAK: tc_barr ≈ 0.2 |

## ξ: two different lengths wearing the same symbol
* **ξ_ribbon** = `coherence_length(p)` — decay length in the SC RIBBON material (region 's').
  Meaningful for GRESTA, where the ribbons are topological: 5.51 sites at the working point, and
  what the `nx ≳ 12 ξ` rule below refers to.
  At the SCHARF point it returns 12.14 sites, which is the TRIVIAL ribbon's evanescent length and
  has NOTHING to do with the end MBS.  **Do not use it there.**
* **ξ_MBS** — decay of the end MBS ALONG the junction, in the normal region.  No function computes
  it; measure it from the real-space LDOS profile.  At the Scharf point ≈ 466 nm ≈ 23 sites
  (3.4068 at y = ±950 nm against 0.4428 at y ≈ 0).
  Every `δ ~ Δe^(−L/ξ)` in the Scharf section means **ξ_MBS**, never `coherence_length`.

## Rules
* Keep existing public names (Params fields, RGFFourTerminal.channels_at_phi, …); scripts depend on them.
* After changing hamiltonians/leads/rgf/transport run `run_all()` and `tests/test_api.py`; all must pass.
* Never let a failed lead solve pass silently (NaN is intended). Report `myf.leads.N_UNRELIABLE` in scans.
* Computation returns arrays; plotting stays in the scripts.

## GRESTA / DOMÍNGUEZ model — dimensionless units, Bz_s in the ribbons
* Working point: nx=80, ny=12, Δ=0.35, α=1.2, tc=1.1, μ_c=−0.03, μ_n=1, μ_s≈0.15, B_z,s≈0.94, η=1e-5 →
  κ=0.497 at k_BT=1e-3 with κ(φ=0)=0.003; connected T→0 plateau in (B_z,s, μ_s).
* Plateau width: ξ_ribbon vs nx (`coherence_length`; nx ≳ 12 ξ_ribbon at T→0, ≳ 7 ξ_ribbon at
  k_BT=1e-3).  Larger Δ and α shorten ξ_ribbon.
* Plateau height: interface matching (tc ≈ 1.1–1.2, μ_n ≈ 1).
* Phase contrast: the centre is depleted (no Dirac point), so the φ=0 gap decays exponentially with ny;
  contrast is lost for ny > 12 (κ(0): 0.006 at ny=12, 0.45 at ny=20). No SOC/Zeeman/mass variant fixes this.
* Dirac model reproduces the paper's Fig. 8 with μ_n ≈ 1 (unstated in the paper).

## SCHARF model (`computations/scharf_ldos.py`; Scharf et al., PRB 99, 214503)
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

## Numerics — measured on the GRESTA point unless stated
(eta-as-absorption is general; the xi-tracking solver limit is about xi_ribbon,
so it applies to GRESTA, where the ribbons are topological.)
* eta is added at EVERY site, so it acts as uniform absorption and costs transmission in
  proportion to path length.  This, not physics, produced the apparent decay of kappa with
  junction length.  Gresta point, beta=0, kT=0, phi=pi:
      eta=1e-5: kappa = 0.4746 / 0.4492 / 0.4252   (nx = 80 / 160 / 240)   <- artefact
      eta=1e-6: kappa = 0.4997 / 0.4944 / 0.4917
      eta=1e-7: kappa =   --   / 0.4991 / 0.4989
      eta=1e-8: kappa =   --   / 0.4996 / 0.4996   <- flat: no decay with length
  The chiral Majorana mode does NOT decay along the junction, as topology requires.
  Use eta >~ 1e-6 for long junctions; eta=1e-5 is only safe for short ones.
* But eta cannot simply be lowered: Sancho-Rubio breaks down when xi is LONG (transfer-matrix
  eigenvalues near 1, so the doubling overflows before converging).  At beta=0.8 (xi=12.2 sites)
  eta=1e-6 gives NaN at nx=80 and overflow warnings at nx=160; beta=0 (xi=5.5) is fine.
  This is a DIFFERENT failure from the documented one (edge gap far below eta).
* Worse, the wrong branch can pass every existing test.  At the Gresta point with eta=1e-7,
  nx=80, kappa(pi)=0.6378 instead of 0.50 with leads.N_UNRELIABLE = 0.  Two candidate detectors
  were tried and BOTH read clean there: the E=0 PHS identity (`myf.phs_residual`, ~5e-13) and the
  retarded-branch condition min eig of Gamma = i(Sigma-Sigma^dag) (positive, +2.5e-10).
  Sancho-Rubio converges to a solution that satisfies the Dyson equation but is the growing
  branch, which neither test distinguishes.  The only signal currently emitted is numpy's
  "overflow encountered in matmul" RuntimeWarning from leads.py:74 -- untested as a flag.
  Practical rule: keep eta >= 1e-6, and distrust any run that printed an overflow warning.
* NOTE T_ee = T_he (ee vs he_cross) is the MAJORANA condition, not a symmetry, so it is NOT
  available as a numerical check.  The genuine identity at E=0 is T_ee = T_hh and T_eh = T_he
  (`phs_residual`, checked in run_all).
* CONSEQUENCE for the Dresselhaus claim below: "larger nx does not help at beta=0.8" was measured
  at eta=1e-5 and is therefore contaminated by absorption (~0.05 of the 0.165 drop at beta=0).
  It cannot be recomputed at eta=1e-6 because the solver fails there for beta=0.8.  Treat the
  nx-dependence at beta=0.8 as UNRESOLVED; the beta=0 baseline (0.4997/0.4944/0.4917 at eta=1e-6)
  is sound.  Re-measured at eta=1e-6 anyway: beta=0.8 gives NaN at nx=80 and
  kappa(pi)=0.993/1.004 at nx=160/240 WITH overflow warnings, against 0.19-0.36 at
  eta=1e-5 -- so the two eta disagree wildly and both are suspect.  Still UNRESOLVED.

## Disorder (onsite, normal region only; `PhysParams.disorder` in meV, `.disorder_seed`)
Scharf point, soc_axis='100', tc_barr_transport=0.2, 32 seeds per cell.  A = (E_Z 0.505, phi=pi)
topological on resonance; B = (0.505, phi=0) and C = (0, pi) trivial.  "LDOS candidate" = Scharf's
own criterion curv_end<0 plus end/mid>3.
* ROBUSTNESS: charge neutrality is far tougher than the transmission magnitude.  At point A,
  |G|/kappa stays at 3e-4 .. 2e-3 from zero disorder up to 3.2 meV (12.8 Delta) while kappa itself
  falls 0.4888 -> 0.2236 (mean) with large scatter.  Neutrality follows from the Majorana
  character, not from clean transmission, so it survives what the magnitude does not.
* Strong disorder DETUNES rather than destroys: at 3.2 meV the mean kappa is 0.22 but the
  per-seed MAXIMUM is still 0.4913.  Individual realisations are pushed off the delta ~ Gamma
  resonance; some land back on it.  LDOS candidates: 32/32 up to 0.4 meV, 31/32 at 1.6, 12/32 at 3.2.
* DISCRIMINATION, and it is NOT via |G|/kappa.  Disorder does manufacture convincing LDOS
  look-alikes at the trivial points -- B gets 2/32 candidates at 1.6 meV and 7/32 at 3.2, C gets
  1/32 and 2/32, with end/mid contrast up to 17-19, BETTER than the genuine MBS (3.2-6.4) at the
  same disorder.  But every one of them is TRANSPORT-DARK:
      B at 3.2 meV: kappa = 2e-7, 4e-7, 5e-8, 7e-8, 2e-9, 2e-7, 7e-6
      C at 3.2 meV: kappa = 5e-12, 4e-13
      A at 3.2 meV: kappa = 0.293, 0.491, 0.425, 0.444, 0.320, ... (12 survivors)
  Six to twelve orders of magnitude.  Reason: LDOS is LOCAL, so one state localized near an end
  gives a zero-energy peak; nonlocal kappa needs a PAIR at both ends with finite overlap delta,
  which is what makes them Majoranas.  Disorder localizes; it does not create end-to-end pairs.
* So the test is TWO-PRONGED, with a measured counter-example for each prong:
      kappa ~ 1/2  rejects disorder-localized trivial states (kappa ~ 1e-7)
      G ~ 0        rejects ordinary charge-carrying channels (the 10 clean-map points, |G|/kappa 0.58-0.98)
  Do NOT read |G|/kappa for a transport-dark state: it is 0/0 and meaningless.
* CAVEAT: the trivial-point candidate counts are 2/32 and 7/32, so the RATE is only known to
  about +-7%.  The orders-of-magnitude kappa gap is safe at 32 seeds; any claim about how OFTEN
  disorder fakes a Majorana needs ~128.  Rare realisations that both look like an MBS and
  transport would be the counter-example that breaks this, and 32 seeds cannot see a 1-in-100 event.
* GOTCHA, fixed in 8129ca0: disorder used to reach neither `LocalGreen` (so the LDOS was blind)
  nor `central_fingerprint` (so scan()/ThermalPoint returned ONE seed's answer for every seed).
  `run_all` now asserts all three paths respond to `disorder_seed`.

## Measurability — can an experimentalist see this?  (conversion: 1 ueV = 11.6 mK)
### SCHARF: no.  Checked three ways.
* The genuine Majorana resonance needs a WEAK probe (tc_barr~0.2) and is then ~0.5-0.8 ueV wide:
  kappa = 0.491 -> 0.44 at kT=0.1 ueV (1 mK) -> 0.019 at 1 ueV.  A fridge delivers 10-20 mK.
* Raising tc_barr to 0.5 widens things and the resonance MOVES in E_Z (to 0.79-0.85 meV) rather
  than vanishing -- so the earlier "strong coupling kills it" was measured at fixed E_Z.  But
  retuning does not rescue it: at L=800/1200 nm the wide kappa~0.5 peaks are CHARGED
  (|G|/kappa = 0.62 / 0.28), i.e. the false-positive class.  Only L=2000 nm stays roughly neutral
  (|G|/kappa=0.060) and there kappa(phi=0)=0.335, so the phase contrast is only 0.16 and kappa
  still halves by kT=2 ueV (23 mK).
* Every configuration is EITHER Majorana-like with no temperature window OR wide but charged.
* The W lever for Scharf is UNTESTED and is NOT the Gresta ny lever: here W sets the Thouless
  energy, E_T = (pi/2)*2*sqrt(mu_N t)/n_across ~ 1/W, and the topological region sits at
  E_Z ~ E_T, so a WIDER junction needs a WEAKER field (W=100 nm -> 2.9 T, W=200 -> 1.4 T,
  W=300 -> 1.0 T).  Opposite direction to Gresta.  Whether that trades field for temperature
  favourably is open.

### GRESTA: yes, but on the feasibility boundary.
* The ceiling is set by the phi=0 MINIGAP, not by the ribbons' bulk gap (0.267 t).  kappa(pi)
  itself survives to kT=0.01 t (0.4581) but kappa(0) rises 0.004 -> 0.236 -> 0.444 over
  kT = 0.001 -> 0.003 -> 0.010 t, so the CONTRAST dies far earlier than kappa(pi) does.
  Measure the contrast kappa(pi) - kappa(0) together with G~0, never kappa(pi) alone.
* The minigap is the hybridisation gap of the two INTERFACE Majoranas across the width, so the
  junction width is a real lever -- and ny=12 is NOT optimal.  Usable ceiling (contrast > 0.40,
  |G| < 0.05), nx=80, eta=1e-6:
      ny= 4: none (kappa(pi) only 0.37-0.41, too few transverse modes)
      ny= 6: kT = 0.003 t     ny= 8: kT = 0.003 t  <- best, kappa(pi)=0.4885
      ny=10: kT = 0.001 t     ny=12: kT = 0.001 t     ny=16: none (kappa(0)=0.34 already)
  So ny=8 TRIPLES the window over ny=12.  There is an optimum, not a monotone trend.
* Physical temperature, anchoring t = Delta_phys/0.35:
      Delta=0.25 meV (Al):  25 mK at ny=8   (8 mK at ny=12)
      Delta=1.0 meV:        99 mK
      Delta=2.0 meV:       199 mK
* THE FIELD IS THE BINDING CONSTRAINT.  Bz_s=0.94 t out-of-plane is 2.3 T for Al-scale Delta,
  against ~0.01-0.1 T for a thin Al film -- an applied field is out by 2-3 orders of magnitude.
  It has to be an EXCHANGE field from a ferromagnetic insulator (EuS/Al gives ~0.1-1 meV, and
  0.94 t = 0.67 meV at Delta=0.25 meV, so it just fits).  Scaling Delta up to get a hotter
  plateau scales the required exchange field the same way (2.69 meV at Delta=1, 5.37 at Delta=2),
  beyond what is demonstrated.  Scale-invariantly: the plateau is visible only for
      k_BT  <~  0.3%  of the Zeeman/exchange energy.
  So the realistic device is Al/EuS at ny=8, giving ~25 mK -- just above fridge base temperature.

### How an experimentalist would actually identify a Majorana (from this work)
1. A 2e^2/h zero-bias peak is NOT sufficient: disorder produced zero-energy end states with
   BETTER LDOS contrast (17-19) than the genuine MBS (3.2-6.4), carrying kappa ~ 1e-7.
2. The decisive measurement is NONLOCAL -- does heat cross from one end to the other.  That needs
   a delocalised PAIR, which is what being Majoranas means, and disorder cannot fake it.
3. Charge neutrality G~0 is the second prong and the robust one (|G|/kappa held at 3e-4 through
   12.8 Delta of disorder while kappa collapsed).
4. BOTH PRONGS CANNOT COME FROM ONE CONTACT.  Strong coupling gives the quantised LOCAL 2e^2/h and
   kills nonlocal transport (kappa=0.001 at tc_barr=1); weak coupling gives nonlocal kappa=1/2.
   The device needs tunable contacts, or two devices.  This constraint is in neither paper.

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
* Dresselhaus hurts the GRESTA plateau in BOTH orientations (distinct from the
  Scharf soc_axis finding above, which is about an in-plane field and trivial ribbons), for different reasons. '110' closes
  the Chern gap through (α−β) and ξ diverges (5.5→49.7 sites for β=0→1.1). '100' keeps ξ short
  (5.5→4.5, since SOC adds in quadrature) but κ(π) overshoots to 0.91–0.99 with G(π)≈0.33, i.e.
  it stops being charge-neutral. Larger nx does NOT help: κ(π) 0.358→0.192 for nx=80→240 at β=0.8.
