# Context for Claude Code — thesis/fourterminal

Master's thesis code: NEGF transport through a four-terminal Josephson junction.
**TWO DIFFERENT SYSTEMS share this package.  Their numbers are NOT comparable — check which
one a statement belongs to before reusing it.**  Current focus: the SCHARF model, with the MIXED SOC
(`tools.mixed_soc(16, 0.15 pi)`) as the thesis study case.  See README.md for structure and usage.

**HOW TO READ THESE NOTES (audited 2026-11-06)** -- sections are dated and grow per topic; a later result beats an
earlier one, and superseded statements are marked SUPERSEDED / LATER in place.  Before reusing a number check its date:
* before 2026-10-13: OLD interface bonds (old tc_barr 0.2 ~ new 0.26; see INTERFACE BONDS) and t = 2.488 meV;
* before 2026-10-14: "mu_N" meant today's mu_C (the CENTRE strip); from 10-14 on mu_N is the PROBE LEADS;
* the 2026-10-07/08 Scharf sections were measured at a = 20 with alpha 14.3 / beta 7.3 ('100') = the mixed SOC;
* "plateau" in the tuning / plateau sections (Nov) = a kappa ~ 1/2 region where eps ~ Gamma (|kappa - 1/2| < 0.03,
  |G| < 0.1 kappa; later also Q = -1 and the two-Majorana relation) -- a RESONANCE held over a range, never a
  protected plateau.  The only protected quantities in Scharf: local Andreev 2e^2/h and det r = -1 (one isolated MBS).

| | Gresta / Domínguez | Scharf |
|---|---|---|
| paper | PRB 114, 125405 | PRB 99, 214503 |
| driven from | `Params` directly | `PhysParams` in `computations/ldos.py` |
| units | dimensionless, t = 1 | physical meV/nm, t = 2.507 meV (CODATA; was 2.488) |
| geometry | nx=80 × ny=12 | nx=100 × ny=5 at a = 20 (L=2000 nm × W=100 nm); a = 10 since Nov (200 × 10) |
| Zeeman | `Bz_s`=0.94, OUT-of-plane, in the RIBBONS | `Bxy_c` in-plane, in the CENTRE; `Bz_s`=0 |
| ribbons | TOPOLOGICAL, Chern = −1 | TRIVIAL, Chern = 0 |
| Majoranas | PROPAGATING chiral mode along the junction | LOCALISED pair at the ends y = ±L/2 |
| α | 1.2 (units of t·a) | 14.3 meV·nm = 0.2865 t·a |
| centre | DEPLETED, μ_c = −0.03 | POPULATED, μ_C = 0.7 meV (called μ_N before 2026-10-14) |
| κ = 1/2 is | a PLATEAU, protected by a bulk gap | a RESONANCE, width = the MBS splitting δ |
| probe contact | transparent: tc ≈ 1.1, μ_n ≈ 1 | WEAK: one bond tc_barr_transport at fixed a (decision 2026-11-02); barriers only in study_rashba.py [a] |

## ξ: two different lengths wearing the same symbol
* **ξ_ribbon** = `coherence_length(p)` — decay length in the SC RIBBON material (region 's').
  Meaningful for GRESTA, where the ribbons are topological: 5.51 sites at the working point, and
  what the `nx ≳ 12 ξ` rule below refers to.
  At the SCHARF point it returns 12.14 sites, which is the TRIVIAL ribbon's evanescent length and
  has NOTHING to do with the end MBS.  **Do not use it there.**
* **ξ_MBS** (= ξ_M) — decay of the end MBS ALONG the junction, in the normal region.  Computed exactly by
  `study_oscillation.slowest_mode` (slowest E = 0 complex-k mode of the cross-section; real-space check in
  the oscillation section, within 1-2 %).  Old LDOS estimate at the Scharf point ≈ 466 nm ≈ 23 sites
  (3.4068 at y = ±950 nm against 0.4428 at y ≈ 0).
  Every `δ ~ Δe^(−L/ξ)` in the Scharf section means **ξ_MBS**, never `coherence_length`.

## Rules
* Keep existing public names (Params fields, RGFFourTerminal.channels_at_phi, …); scripts depend on them.
* After changing hamiltonians/leads/rgf/transport run `run_all()` and `tests/test_api.py`; all must pass.
* Never let a failed lead solve pass silently (NaN is intended). Report `myf.leads.N_UNRELIABLE` in scans.
* Computation returns arrays; plotting stays in the scripts.

## FORMULAS: see DERIVATIONS.md (2026-11-06; 2026-11-18: section 7.7 poles / EPs added, all code links re-synced -- each link checked to hit the code it named; DERIVATIONS.pdf NOT rebuilt: no pandoc on this machine)
Every computed quantity (model, leads, RGF, channels, kappa, G, Andreev, det r, two-Majorana relations, Gamma, Q,
xi_M, eps, overlap law, end modes, LDOS): equation, derivation, physics, verified literature source (equation
numbers checked against the arXiv LaTeX/PDF) and the code lines.  Keep it in sync when a formula changes, and when
code lines move (the links name line numbers).  The .md is the source: after editing it, rebuild DERIVATIONS.pdf with
`py derivations_pdf.py` (pandoc + lualatex; the PDF's code links are vscode:// links, its § and arXiv IDs are
clickable), and keep display equations within the A4 width (break long ones with aligned/gathered).  Found while
writing it: Gresta et al. Eq. (4) as printed lacks the 3/pi^2 of its own T -> 0 limit (the code uses the normalised
kernel); the studies evaluate channels R -> L ('left'), which equals L -> R for kappa at E = 0 (unitarity + PHS).
Lewenkopf-Mucciolo Eq. (91) (decimation surface term) has a typo, eps_{k-1} for eps^s_{k-1}; leads.py uses the
correct recursion.  Tags (2026-11-06): [source, verified] / [follows from source] (algebra on verified equations) /
[derived here] (code conventions or own analysis, no paper exists) / [model choice].

## CODE LAYOUT of computations/ (merged 2026-11-06 -- keep it small)
| file | parts (former scripts) |
|---|---|
| `tools.py`, `ldos.py` | shared functions; PhysParams |
| `study_rashba.py` | W, wbands, delta, tc, checks, T, transition, soc_shift, fermi (check_fermi), a -- pure Rashba, mostly a = 20 (wbands: mixed vs Rashba bands over W) |
| `study_oscillation.py` | oscillation, branches, fingerprint -- eps(E_Z), pure Rashba a = 20 |
| `study_tuning.py` | topology, coupling, tuning, maps, leads, spectrum, ep, ep2, ep3, ep4 -- a = 10 couplings / topology / survey maps; a = 5 (E_Z, phi) maps; mixed SOC vs tc and tau (+ checks: tc > 1, tau 1 charge); DOS(E, phi) vs probe bond; exceptional points of the open junction (tools.nh_spectrum) |
| `study_mixed.py` | oscillation, plateau, amplitude, deck, combine, angle -- the mixed SOC (thesis case): eps(E_Z), the best plateau, the end-mode amplitude, the advisor-deck runs, the Gresta + Scharf combination checks, the SOC angle and field rotation |
| `slides_nov.py`, `figures.py` | advisor-deck figures, Scharf paper figures (own parameters).  Disorder: study_tuning [dis] / [cdis] (disorder.py, which took the ldos.PP sandbox, was DELETED 2026-11-25) |
* `python study_mixed.py plateau` runs one part (several: list them; none: all, in order).  `run_part('part')`
  replaced the `if __name__ == "__main__":` guard; cells are tagged `# %% [part]`.  References below read
  "file [part] cell N".  NEVER bind run_part / PARTS / _SELECTED / sys inside a part: the first version called the
  selector run() and the plateau cells' own variable `run` overwrote it (crashed the plateau figure cell; caught by the
  cache-only smoke run, fixed by the rename).
* Renamed on merging (module-level names that collided): rashba CASES -> CASES_D / CASES_TC / CASES_SOC, tag -> tag_D,
  WIDTHS -> WIDTHS_TR (transition); oscillation EZ -> EZ_B (branches), EZ_F (fingerprint); tuning junction ->
  junction_topo / junction_sn, EZ -> EZ_TOPO / EZ_SN, TAUS -> TAUS_SN, PHI -> PHI_SN; mixed EZ -> EZ_AMP.
  `study_tuning.junction` and every cache name are unchanged.  Verified: per-part AST identical to the originals
  with these renames, every module-level object and every global a function reads identical, no header name rebound;
  a cache-only smoke run of every part (run_scan refuses to compute) passes, and the mixed parts reprint
  logs/oscillation_ / plateau_ / amplitude_mixed.log line for line (every line of logs/study_oscillation, study_T, study_a,
  study_transition reappears too).  rashba [checks] computes end-LDOS splittings directly (uncached, ~8 min/point, as
  the original study_checks.py did) -- slow, not broken.
* NEW WORK GOES INTO THE MATCHING FILE AS A NEW PART OR CELL, not into a new study_*.py.  Watch module-level name
  collisions: the parallel workers import the whole file and see the LAST definition of a name.
* Archived (computations/archive/): soc_bond_compare.py, mu_scan.py, ftc.py, early_prototypes/ (the old computations/tests).

## GRESTA / DOMÍNGUEZ model — dimensionless units, Bz_s in the ribbons
* Working point: nx=80, ny=12, Δ=0.35, α=1.2, tc=1.1, μ_c=−0.03, μ_n=1, μ_s≈0.15, B_z,s≈0.94, η=1e-5 →
  κ=0.497 at k_BT=1e-3 with κ(φ=0)=0.003; connected T→0 plateau in (B_z,s, μ_s).
* Plateau width: ξ_ribbon vs nx (`coherence_length`; nx ≳ 12 ξ_ribbon at T→0, ≳ 7 ξ_ribbon at
  k_BT=1e-3).  Larger Δ and α shorten ξ_ribbon.
* Plateau height: interface matching (tc ≈ 1.1–1.2, μ_n ≈ 1).
* Phase contrast: the centre is depleted (no Dirac point), so the φ=0 gap decays exponentially with ny;
  contrast is lost for ny > 12 (κ(0): 0.006 at ny=12, 0.45 at ny=20). No SOC/Zeeman/mass variant fixes this.
* Dirac model reproduces the paper's Fig. 8 with μ_n ≈ 1 (unstated in the paper).

## INTERFACE BONDS SCALE AS A WHOLE (fixed 2026-10-13) -- read before trusting older numbers
* Before the fix, model_hop(p, dir, tc, alpha=bond_alpha) scaled only the -tc tau_z part of an
  interface bond; the Rashba/Dresselhaus part stayed at full strength.  At the Scharf point that is
  a 0.16 t SPIN-FLIP hop surviving tc_barr = 0: kappa = 0.208 at point A with "decoupled" probes.
* Now every interface bond (probe: tc_barr, ribbons: tc_top/tc_bot) is model_hop(..., bond=True),
  built from the global alpha, beta:  tc <= 1: tc x [-tau_z + SOC] (a barrier lowers the tunnelling
  amplitude, not the spin rotation per hop);  tc > 1: -tc tau_z + SOC (better matching enhances the
  hopping, NOT the SOC, a material property).  Continuous at tc = 1.  So the GRESTA point (tc = 1.1)
  is exactly as before: kappa(pi) = 0.4997 at eta = 1e-6, kT = 0.  (A brief version that also scaled
  SOC above 1 gave 0.486 there -- the tc > 1 cap is what keeps the Gresta numbers valid.)  No averaging over the two regions (per-region alpha_c/
  alpha_n/alpha_s and t_n/t_c/t_s do NOT enter the bond; tc is the bond hopping in units of the unit
  lattice hop).  tc = 1 is bit-identical to a bulk bond; Dirac unchanged.  The spin rotation per hop
  (9.07 deg at Scharf) and its axis are exactly the bulk ones for tc <= 1; above 1 the angle shrinks
  (8.25 deg at tc = 1.1), the axis stays.  model_hop raises unless t_n = t_c = t_s = 1, because the
  1.0 in min(tc, 1) IS the lattice hop.  run_all 28/28, tests/test_api.py 5/5.
* What it changes (computations/archive/soc_bond_compare.py keeps the OLD variant, scaled=False):
  - OLD tc_barr = 0.2 behaves like NEW tc_barr ~ 0.26 = sqrt(0.2^2 + 0.16^2): the clean E_Z sweep
    reproduces the old resonance positions and heights.  So every Scharf weak-probe number below
    measured at "tc_barr = 0.2" (or 0.5) belongs to a stronger coupling than its label.
  - NEW tc_barr = 0.2 moves the resonances (0.43, 0.485, 0.53, 0.555, 0.625, ... meV); point A
    (E_Z 0.505) then sits on a flank, kappa 0.402.  PP's working point needs re-tuning.  [LATER: ldos.PP is no longer a working point -- since 10-22 it is an a = 10,
    W 150, L 500, E_z = 0 sandbox; the studies use tools.base_params / case and their own junction(); see disorder below.]
  - Qualitative claims all survive: quantised maxima 0.45-0.49, G ~ 0 (|G|/k <= 0.023 at peaks),
    |G|/kappa still separates the false positives on the 61x61 map (98% of kappa>0.05 neutral),
    weak disorder harmless, strong-probe results identical.
  - Scharf's LDOS (tc_barr = 0) is now truly decoupled; end LDOS at A 10.42 -> 10.58.
* PHYSICAL TUNNEL CONTACT (2026-10-13), no rescaling at all: Params.barrier_n / barrier_V
  (units of t) put barrier_n cells of lead material (same t, alpha, beta) at +barrier_V tau_z
  between each probe and the centre, folded EXACTLY into the probe self-energy (Lead(barrier=...),
  one Dyson step per cell; geometry unchanged).  Use with tc_barr = 1, so every bond is a bulk
  bond.  barrier_where='junction' instead raises the first/last barrier_n columns of the centre
  (enters via column_onsite, so RGF, LDOS and now the dense H_C all see it).
  Scharf point A (E_Z 0.505), phi=pi (old constants, t = 2.488 meV):  Tr Gamma_L falls exponentially with V_b and n_b, and
  kappa sweeps the resonance --  n_b=1: kappa 0.000/0.010/0.152/0.484 at V_b = 1/2.5/5/10 meV
  (Andreev 2.00 -> 0.36);  n_b=2: 0.004/0.410/0.085/0.001 (Andreev 1.99/0.99/0.004/0);
  G <= 3e-4 throughout.  So the complementarity local Andreev <-> nonlocal kappa now comes from a
  real barrier.  Checks: run_all 28/28 (lead material as barrier == no barrier; RGF == dense with
  junction barrier + disorder; Tr Gamma falls with V_b).
  Dense H_C previously IGNORED disorder (only RGF/LDOS used column_onsite); fixed with the above.
* STALE DATA (written 10-13): computations/plot_npz/scharf_disorder_*.npz were computed with the OLD bonds.  Their
  kappa, G, andreev (weak probe) are pre-fix; kappa_s, G_s, andreev_s, det are unaffected.
  disorder.load_or_run will return them without noticing -- delete them to recompute.
  [AUDIT 2026-11-06: only two disorder caches remain, both written 2026-10-14 -- weak_strength at the MOVED point
  A = (0.98, pi), so after the bond fix (inferred from date and field, not re-verified), and weak_trivial_B.  The
  numbers in the Disorder section (A = 0.505, old bonds) are in NO current cache.  And disorder.py builds its
  parameters from ldos.PP, which is now the a = 10 / W 150 / L 500 sandbox: running disorder.py today computes a
  DIFFERENT junction (clean A gives kappa 0.140, the cache 0.413).  Pin its parameters before reusing it.]
  [2026-11-25: disorder.py DELETED; the two scharf_disorder_* caches remain, no code reads them.]

## SCHARF model (`computations/ldos.py`; Scharf et al., PRB 99, 214503)
* MATCH TO THE PAPER (checked line by line against Eq. 1 / App. B Eq. B6, 2026-10-13): basis, kinetic
  term, the m(alpha^2+beta^2)/2hbar^2 shift, the [100] SOC hops, pairing sign, Zeeman (theta_code =
  pi - theta_Z), mu_N/mu_S, a = 20 nm, 5 x 100 N sites -- all EXACT.  Deliberate differences, kept:
  semi-infinite S ribbons (Scharf: W_S = 450 nm hard walls; phase boundaries barely differ, App. A2),
  Lorentzian eta (Scharf: Gaussian Gamma = 0.05 Delta on eigenstates), the transport probes (not in
  Scharf), and the phase on the TOP ribbon = Scharf's x < -W/2 side, i.e. phi -> -phi, compensated by
  the + pi on theta_z in PP (symmetry phi -> -phi with E_Z -> -E_Z).  Constants are CODATA since
  2026-10-13 (t 2.488 -> 2.507 meV, E_T = 832 ueV exactly); N_MID = 5 (100 nm bulk probe, Fig. 8g).
  Caches written before 2026-10-13 (scharf_kappa_map_*, scharf_mu_*, scharf_soc_*) predate the constants change
  (0.8% in t); everything later uses CODATA.
* BOND SETTINGS THAT REPRODUCE SCHARF (keep these):
  - tc_top = tc_bot = 1: Scharf's S/N interface is a plain bulk bond; the mismatch is mu_N vs mu_S
    only.  With the whole-bond scaling, tc = 1 is bit-identical to the bulk bond, SOC included.
  - LDOS: tc_barr = 0 = Scharf's hard walls at y = +-L/2.  Only exact WITH the whole-bond scaling
    (the old bond left a 0.16 t spin-flip hop to the probes).
  - Transport (our addition): tc_barr < 1 is a reduced-transparency contact whose SOC scales with it
    (old tc_barr 0.2 ~ new 0.26).  For a contact made of the material itself use tc_barr = 1 with
    barrier_n / barrier_V.  Either way: barrier_V = 0 and tc_barr = 0 for anything meant as Scharf.
  - alpha_c / alpha_n / alpha_s = None (one global alpha, beta everywhere, as in Scharf).
* [The bullets from here to the end of this section are 2026-10-07: OLD bonds, t 2.488, alpha 14.3 / beta 7.3, a 20.
  Qualitative claims re-checked 10-13 (INTERFACE BONDS); numbers not re-measured.]
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
  κ = 0.491 → 0.44 at k_BT=0.1 μeV (≈1 mK) → 0.019 at 1 μeV.  [SUPERSEDED as a verdict: one field, old tc 0.2.
  With the full field scan and tc 0.35-0.4 neutral κ ~ 0.48 survives 20 mK (Measurability, 2026-10-21/23).  Still a
  resonance: the width is Γ, not a gap.]
* |G|/κ DISCRIMINATES genuine Majoranas from LDOS false positives, and is resonance-INdependent:
  T_ee=T_he holds to 4–5 digits everywhere the Majorana exists, not just on resonance. On the
  61×61 '100' map, of 1064 points with κ>0.05, 1011 are neutral (|G|/κ<0.1) and 10 are charged
  (>0.5). All 10 pass Scharf's own curv_end<0 criterion, yet have end/mid contrast 1.7–3.3
  (vs 7.7–13.0 for the genuine ones) and sit at E_Z ≳ 0.9 E_T — delocalized zero-energy states.
  Caveat: |κ−0.5| alone is a valid POSITIVE only; a genuine Majorana off resonance also gives κ≈0.
  Maps cached in `computations/plot_npz/scharf_kappa_map_{100,110}.npz` (61×61, ~16 min each).
* soc_axis matters: '110' puts (α−β) on the propagation axis, '100' puts √(α²+β²). With β=7.3
  that halves the effective SOC at '110' — κ>0.05 on 9.6% of the map vs 28.6% at '100'.

## SCHARF: probe leads, mu_N vs mu_C, lattice constant (2026-10-14/19)
* NAMING CHANGED: in `PhysParams`, `mu_N` is now the PROBE LEADS (Params.mu_n) and `mu_C` the
  normal strip between the SCs (Params.mu_c; Scharf's "N").  `mu_C=None` means mu_C = mu_N (the
  old single-mu behaviour); `pp.mu_centre` returns the effective value and E_T / alpha k_F use it.
  EVERY "mu_N" written before 2026-10-14 -- the sections above AND the 10-07/08 Measurability, Andreev-plateau,
  'WHY THERE IS NO kappa PLATEAU' and (E_Z, mu_N)-wedge sections further down (0.7 meV working point, the 0.65-0.95
  wedge, the subband edge) -- means the CENTRE, i.e. today's mu_C (then one mu set leads AND centre).  From 10-14 on
  (e.g. MORE KNOBS AT FINITE T: mu_N 1-5.5) mu_N is the probe leads.  ldos.PP: mu_N = 3.0 (leads), mu_C = 0.7.
* WHY: the probes are W-wide hard-wall strips; at mu = 0.7 meV they sit at their first transverse
  subband (lattice 2t(1-cos(pi/(ny+1))): 0.672 / 0.812 / 0.896 meV at a = 20 / 10 / 5 nm,
  continuum ~0.99 at W = 100).  Open at a = 20 only by 0.03 meV -- a lattice artefact; CLOSED at
  a = 10 and 5 (kappa = 6e-11, Andreev 2e-12: a dead contact).  Opening the leads (mu_N = 3) at
  a = 5, L = 300 restores transport (kappa 0.73-0.78, but G ~ kappa: L < xi_MBS, transparent probe).
  So the "kappa = 0 below mu_N ~ 0.62" edge in the wedge section is the PROBE closing, not the strip.
* tc_barr / tc are DIMENSIONLESS (fraction of the lattice hop) -- never divide by t.  But a fixed
  tc_barr is not a fixed contact: one weak bond has T ~ 4 tc^2 (k_F a)^2, so it gets ~a^2 more
  opaque on refining the lattice.  An a-independent contact = physical barrier (V_b meV -> barrier_V
  = V_b/t, d nm -> barrier_n = round(d/a), tc_barr = 1) or the mu_N/mu_C density step.  [Implemented 2026-10-23 as
  PhysParams.V_barr / d_barr and lam_barr; used only in study_rashba.py [a] since the 11-02 contact decision.]  tc = 1 (S/N) is a bulk bond and a-independent.
* eta_rel_transport = 1e-5 Delta is 6e-8 t at a = 5 nm, below the eta >= 1e-6 t solver rule; no
  failed solves seen, but untested territory.  [TESTED 2026-11-11, see "SCHARF: lead solver and eta" under Numerics:
  correct branch, kappa within 0.001 of eta -> 0.]
* LATTICE CONSTANT (pure Rashba, cross-section, phi = pi): gap over all k at W = 100:
  66.66 / 66.76 / 66.75 ueV for a = 20 / 10 / 5 -- converged.  At W = 300: 33.9 / 29.1 / 28.2 ueV
  (E_Z 0.323) -- a = 20 is 15-20% off, a = 10 within 3% of a = 5.  Q and the ABS count never change.
  kappa resonance POSITIONS shift with a by ~0.02-0.03 meV (phase k_F L, worse at larger L);
  heights and neutrality do not.

## SCHARF: more ABS by widening W (2026-10-19/20) -- it does NOT help kappa
CODE (reorganised 2026-10-22): `computations/tools.py` holds everything shared, as plain
functions (base_params/case, cross-section H(k), Pfaffian Q, gaps, transport_point, channel_point +
thermal, LDOS, run_scan with functools.partial; save_fig writes computations/plots/<name>.png).  [The file layout
below is OUT OF DATE -- see CODE LAYOUT at the top (merged 2026-11-06).]  Thin studies import ONLY it: `study_rashba.py` [W]
(ABS count, gaps, k_x = 0, optional ldos map), `study_rashba.py` [checks] (a, double peak,
long L), `study_rashba.py` [delta] (+ 4 ABS at long L), `study_rashba.py` [tc], `study_rashba.py` [T]
(finite T, tc at finite T), `study_rashba.py` [transition] (LDOS transition, unfinished).  The old
scharf_abs*.py scripts and logs are in computations/archive/ (reference only; they no longer run
from there).  base_params() writes the pure-Rashba set out explicitly -- it no longer follows
ldos.PP.  Caches plot_npz/scharf_abs*_*.npz keep their old names (verified identical) and
do NOT encode parameters: delete them if parameters change.  The finite-strip (H_device) code was dropped;
it lives only in archive/scharf_abs.py.  (The mixed SOC came back as tools.mixed_soc -- the thesis case.)
* INVARIANT without transport: infinite junction along the interfaces, cut across with 900 nm SC
  strips + hard walls (Scharf: 450; ABS near Delta converge only from ~900).  Q = sign Pf M(0) Pf M(pi)
  in the Majorana basis (Pf^2 = det checked).  Agrees with det r: Q = -1 at (0.5, pi), +1 at E_Z = 0.
* ABS COUNT (distinct levels < 0.9 Delta at E_Z = 0, k = 0, phi = 0) set by Delta/E_T, independent
  of SOC:  W = 100: 1,  200-500: 2,  800: 3,  900-1000: 4  (E_T = 0.83 ... 0.08 meV).
* WIDENING CLOSES THE GAP AT FINITE k (trajectories ~ parallel to the interfaces), not at k = 0.
  Gap over all k at the best topological point, phi = pi:
      W [nm]   100     300     800     1000
      beta!=0  38      8       1.7     2.5   ueV   (alpha 14.3, beta 7.3, '100')
      beta=0   67      34      10      --    ueV   (alpha 16)   <- pure Rashba 2-6x better
  Q = -1 still holds over a large (E_Z, phi) diamond starting at E_Z ~ 0 (lower field: a plus).
  [CORRECTED 2026-11-11, study_rashba.py [wbands], plots/W_bands_mixed.png: the table above is the 88-point K_GRID
  minimum, which misses the narrow finite-k dips.  Refined at EVERY field of the same lines (313-point grid + 1D
  refinement), best topological field per W:
      W [nm]             100     300     800     1000
      mixed  Delta_top   37.2    0.53    0       0      ueV   gap CLOSED (< 0.01 ueV) on 0/40, 17/35, 40/40, 35/35
                                                              of the Q = -1 fields
      Rashba Delta_top   65.8    25.1    2.5     --     ueV   never closed (0/40, 0/37, 0/38)
  So for the MIXED SOC (thesis case) only W = 100 nm has a gap at all; from W = 800 the infinite junction is gapless
  at finite k everywhere in Q = -1 (Q still -1: it only probes k = 0 and pi).  The bands show the mechanism: W 100
  one Andreev band whose minimum (37 ueV, k 0.034/nm ~ the outer Fermi point) is far below its k = 0 value (138);
  W 300 / 800: 2 / 3 ABS, denser bands, dips at ~0.018 and ~0.034/nm that reach zero.  Why the mixed SOC closes and
  pure Rashba keeps a small gap is NOT worked out.]
* RESULT, pure Rashba, weak probe tc_barr = 0.2, phi = pi:
  - W = 100 (1 ABS), L = 2 um: genuine.  delta 0.7 ueV, next level 74 ueV, LDOS end/mid 23,
    kappa peaks 0.49-0.50 with G ~ 0 (resonances only above E_Z ~ 0.6: below, delta << Gamma).
  - W = 300 (2 ABS), L = 2 um: NOT separated (delta 24 vs E_1 28 ueV, flat LDOS), peaks 0.55-0.58
    CHARGED (|G| 0.2-0.3).  L = 4 um: lowest level 11.5 -> 4.5 ueV, end/mid 0.8 -> 2.2, and EIGHT
    neutral peaks 0.45-0.498 (|G|/kappa 0.03-0.06); a = 10 nm gives the same (0.494/0.495, |G|/kappa
    0.03-0.05) at shifted E_Z.  L = 8 um: |G|/kappa 0.002, peaks 0.497 -- but narrower (grid misses
    most).  xi_MBS ~ 2 um at W = 300 (vs ~0.25 um at W = 100), from 11.5 -> 4.5 ueV over 2 um.
  - W = 800 (3 ABS): kappa 0.7-1.0, partly charged at L = 2 and 4 um; at L = 8 um the NEUTRAL peaks
    (|G|/kappa <= 0.01, E_Z 0.20-0.24) are 0.56-0.68 > 1/2: several neutral near-zero channels.
    Length does not remove them (bulk finite-k modes, not end overlap).
* THE DOUBLE PEAKS (kappa ~ 1/2, dip, 1/2) are the delta = 0 node of delta ~ cos(k_F L):
  W = 300, L = 4 um, fine scan: left peak kappa 0.5015 (delta/2 = 1.0 ueV, Andreev 0.56), dip 0.150
  (delta/2 < 0.2 ueV unresolved, Andreev 1.81), right peak 0.5000 (0.75 ueV, Andreev 0.59).
  kappa is evaluated AT E = 0 (T -> 0), so this is not an energy-grid effect; the dip does not reach
  0 on a 0.001 meV grid (node narrower than the step).  Peaks are ~0.005 meV wide in E_Z at 4 um:
  a 0.0035 meV grid under-samples heights -- use fine scans before quoting a height.
* So: more ABS trades the protecting gap and the single-channel character for a lower field.
  Usable for quantised heat: 1-2 ABS (2 ABS needs L >~ 4 um).
* LARGER DELTA DOES WHAT WIDENING CANNOT (2026-10-20, `study_rashba.py` [delta], pure Rashba, a = 20,
  L = 2 um, phi = pi).  The gap over all k is set by W, not Delta or SOC:
      W = 100:  Delta 0.25 / 0.5 / 0.75 / 1.0 meV -> 1 / 2 / 2 / 2 ABS, gap 67 / 71 / 68 / 65 ueV
      W = 300:  Delta 0.25 / 0.5 / 0.75 / 1.0 meV -> 2 / 2 / 4 / 4 ABS, gap 34 / 32 / 31 / 29 ueV
      W = 300, alpha 32 (x2): gap 31 ueV (Delta 0.25), 31 (0.5) -- stronger SOC does NOT help.
  Transport (neutral = |G|/kappa < 0.05):
  - W = 100, 2 ABS, L = 2 um: GENUINE.  Delta 0.5: 9/11 peaks neutral, 6 at 0.48-0.498.
    Delta 0.75: 16/19 neutral, 8 at >= 0.48.  Delta 1.0: 20/23 neutral, 10 at >= 0.48 (0.49-0.493).
    Resonances get DENSER with Delta and start at low field (0.06-0.1 meV); G grows above ~0.9 meV.
  - W = 300 at L = 2 um fails for EVERY Delta and alpha (kappa 0.5-0.99, |G|/kappa 0.13-0.76): its
    ~30 ueV gap needs L >~ 4 um, whatever the ABS count.
  - 4 ABS (W = 300, Delta 0.75/1.0), `study_rashba.py` [delta] cell 3: L = 4 um still CHARGED (peaks 0.44-0.66,
    |G|/kappa 0.17-0.36 -- where 2 ABS at Delta 0.25 was already neutral).  L = 8 um (Delta 0.75):
    GENUINE -- 14/17 peaks neutral, 0.46-0.49 (5 at >= 0.48), δ=0 double peaks, and NO neutral
    peak above 0.52 (single channel; contrast W = 800, 3 ABS, 8 um: neutral 0.56-0.68).
    So more ABS at fixed gap still lengthens the required L (2 ABS: 4 um, 4 ABS: 8 um at W = 300).
  - Caveat: Delta >= 0.75 approaches mu_S = 1 meV (strong-coupling SC); treat as a trend.
  Rule: ABS count ~ Delta/E_T, protecting gap ~ f(W); raise Delta for ABS, keep W small for the gap.
* k_x = 0 (normal incidence) vs ALL k (2026-10-20, `study_rashba.py` [W] cell 3, pure Rashba, a = 20):
      W [nm]               100          300           800
      ABS                  1            2             3
      Q = -1 at phi = pi   0.02-1.25    0.035-0.42    0.005-0.19 and 0.21-0.30 meV
      max gap at k_x = 0   170          128           76     ueV
      gap over all k       67           34            10     ueV   (ratio 2.5 / 3.8 / 7.6)
      [refined 2026-11-11: 65.8 / 25.1 / 2.5 ueV at the refined best fields -- see [wbands] above]
  Diamond centre at phi = pi = E_T (0.71 / 0.25 / 0.10 meV vs E_T 0.83 / 0.28 / 0.10); period in
  E_Z = 2 E_T, so W = 800 shows a checkerboard (second diamond from E_Z ~ 0.2).  Semi-infinite
  ribbons (junction_spectrum_kx) reproduce the 900 nm strip levels at W = 300.  Andreev
  quantisation explains all of it: N ~ W/(pi xi) + 1, dE ~ pi hbar v_F/(W + xi), gap(k_x=0) <~ dE/2,
  Delta_top ~ hbar^2/mW^2 (grazing modes; Scharf Sec. III D) -- k_x = 0 sets the phase boundaries,
  finite k_x sets Delta_top, and the two drift apart with W.
* THE PROBE COUPLING DOES NOT REPLACE LENGTH (2026-10-20, `study_rashba.py` [tc], tc_barr_transport
  0.05/0.1/0.2/0.5/1, phi = pi; Majorana-like = 0.45 <= kappa <= 0.55 and |G|/kappa < 0.05):
      W 100 L 2 (1 ABS):        3 / 3 / 3 / 0 / 0   Majorana-like peaks
      W 300 L 4 (2 ABS):        0 / 1 / 6 / 0 / 0
      W 300 L 2 (2 ABS):        0 everywhere; kappa up to 2.5, |G|/kappa 0.24-0.67 as tc grows
      W 800 L 4 (3 ABS):        0 everywhere; kappa > 1/2 already at 0.05, up to 7 at tc = 1
      W 300 D 0.75 L 4 (4 ABS): 0 everywhere; kappa ~ 0.5-0.7 at 0.5 but |G|/kappa ~ 0.2
  tc only moves Gamma; a clean pair needs delta << E_1, which only L (or a larger Delta_top)
  provides.  Best tc is 0.1-0.2 even in the working cases; >= 0.5 couples the higher ABS (G != 0).
  det r < 0 over most of the field range at tc = 1 also in FAILING cases (kappa 2.5): its sign is
  meaningful only together with neutral kappa = 1/2 or a quantised local Andreev.
* LDOS THROUGH THE TRANSITION (2026-10-23, study_rashba.py [transition], L 2 um, phi pi, eta 0.05 Delta):
  W 300: inside Q = -1 the MIDDLE LDOS is gapped at E = 0 (a dark diamond following the k = 0 gap) while
  the END keeps zero-energy weight; real-space LDOS(E = 0) end/middle ~ 2.5 at the best point (0.323 meV).
  W 800: the middle shows the same checkerboard of levels as the end and the LDOS(E = 0) profile is flat
  (end/middle ~ 1.2): no end localisation.  NB eta = 12.5 ueV exceeds Delta_top at W 800 (10 ueV), so
  Scharf's broadening cannot resolve it there anyway; the 'trivial' and 'gap closing' picks of W 800
  both fall on E_Z = 0 (the first topological field is the second grid point).

## SCHARF: lattice-constant convergence and the probe contact (2026-10-23/24, `study_rashba.py` [a])
* SCALING: PhysParams.params() divides every energy by t = hbar^2/2ma^2, alpha by a t, W and L into
  sites; mu_S/N/C, Delta, alpha, E_Z, eta, W, L, E_T map back identically for a = 20 ... 2.5 nm.
* EXCEPT tc_barr: it scales ONE lattice bond, and a weak bond's transmission falls like a^2 (1D:
  tc_barr 0.2 -> T = 0.074 / 0.023 / 0.006 / 0.0016 at a = 20 / 10 / 5 / 2.5 nm).  Every earlier a-test
  with fixed tc_barr compared DIFFERENT contacts (a = 10 had ~3x less coupling).  Use the physical
  barrier instead: PhysParams.V_barr (meV) over d_barr (nm, multiple of a) -> my_functions barrier_V /
  barrier_n (lead material raised by V, folded into the probe), with tc_barr_transport = 1.
  1D T: 6 meV x 20 nm ('medium', ~ tc 0.4) 0.323 -> 0.296; 6 meV x 40 nm ('weak', ~ tc 0.2) 0.049 -> 0.039.
* CROSS-SECTION converged at a = 20: Delta_top 66.66 / 66.76 / 66.75 / 66.75 / 66.75 ueV (E_Z 0.593),
  gap(k=0) 152.7 -> 150.0 ueV (settled by a = 5); 1 ABS at every a.
* TRANSPORT (W 100, L 2 um, phi pi, E = 0): kappa heights (~0.50) and the neutral/charged pattern
  (neutral below ~0.95 meV, |G|/kappa 0.1-0.7 above, both barriers) are the same at every a.
  PEAK POSITIONS move by up to 0.05-0.07 meV from a = 20 to 5 (cos(k_F L) phase: k_F a = 0.53 at
  a = 20 gives a ~1% k_F error = ~0.6 rad over 2 um) and converge like a^2 below 5 nm:
      medium, neutral peaks:  a 5: 0.794 0.838 0.904   a 4: 0.792 0.842 0.906   a 2.5: 0.790 0.850 0.908 meV
      weak (a 5 vs 4): 0.572/0.578 0.616/0.616 0.724/0.726 0.736/0.738 0.876/0.880 0.882/0.884
  Rule: a = 20 for physics scans; a <= 5 when the exact resonance fields matter.  Majorana-like peak
  COUNTS near the 0.45 / 0.05 cuts flip between a values -- compare positions, not counts.
* Cost per transport point ~ a^-4 (0.2 s / 4 s / 31 s / ~80 s / ~500 s); a = 2.5 needs several GB per
  worker: 14 workers ran out of 32 GB -> transport_line(..., n_workers=4).
* ONE-SITE (delta) BARRIER, PhysParams.lam_barr = V a fixed (one site of lam/a; cell 6, tc_barr = 1):
  converges too, heights ~0.50 and neutral pattern at every a:
      site120 (lam 120 meV nm): a 20: 0.756 0.862 0.916 | a 10: 0.812 0.910 1.002 | a 5: 0.830 0.928 1.032
                                | a 4: 0.834 0.930 1.036 meV
      site240 (lam 240):        a 10: 0.684 0.760 0.868/0.892 | a 5: 0.684/0.712 0.774 0.892/0.914
                                | a 4: 0.684/0.718 0.776 0.896/0.916 meV
  The a = 20 -> 5 shift is larger (~0.07 meV) than for the square barrier: at a = 20 the "one site" is a
  20 nm wide barrier, only at small a a true delta.  Converged positions differ from 'medium' (same
  V d = 120) because 6 meV x 20 nm is not thin (k_F d ~ 0.7) -- different contacts, both converge.
* THE CONTACT FROM THE FULL MODEL (2026-11-02, study_rashba [a] cell 7, study_rashba.contact_scattering): left-probe
  reflection matrix r_ab = -delta_ab + i Gamma_a^1/2 G_LL[a,b] Gamma_b^1/2 (e/h blocks, open channels only),
  normal state = Delta 0 with everything else unchanged.  Validated in the SC state (a 20, tc 0.2, E_Z 0.6,
  pi): R_he = eh_local exactly, det r agrees; n_e - R_ee - R_he = 0.1113 vs right-probe ee + he_cross 0.1093,
  the 0.002 gap = absorption by eta (2 eta / Gamma ~ 2 x 2.5 neV / 2.3 ueV).
      contact         a    open e-ch   T_out    per channel
      tc_barr 0.2     20      4        0.090    0.041 0.041 0.004 0.004
      tc_barr 0.2     10      2        0.023    0.0115 x2
      tc_barr 0.3387  10      2        0.074    0.037 x2
      lam 240         20      4        0.103    0.048 0.048 0.0035 0.0035
      lam 240         10      2        0.091    0.045 x2
  - The 1D chain numbers above (cell 2: "T = 0.074 / 0.023 / ...") are NOT the probe's transmission: the
    real per-channel value is ~half (0.041 at a 20), and the 1D model has no channel count.
  - LATTICE ARTEFACT IN THE PROBE: at a = 20 the 5-site lead has TWO transverse subbands open at mu_N = 3
    (0.67 / 2.49 meV), at a = 10 only one (0.81 / 3.18); continuum (W 100) 0.99 / 3.96 -> one.  The second
    subband carries ~10% of T_out at a 20.
  - tau = 0.3387 (matched in 1D) gives 10% LESS per channel at a 10 than tau 0.2 at a 20; lam 240 changes
    by 6%.  Whether equal per-channel T means equal Majorana Gamma is NOT tested (needs kappa(E_Z) lines).
* DECISION (2026-11-02): the probe contact is ONE weak bond tc_barr_transport at a fixed lattice constant.
  All barrier models (lam_barr, V_barr/d_barr), the 1D chain contact_1d and contact_scattering live ONLY in
  study_rashba.py [a] (the a-convergence check); new studies use tc_barr_transport and no barrier.
* COUPLING = BROADENING: kappa/kappa_0 = 2x^2/(1+x^2)^2, x = eps/Gamma -> height 1/2 independent of the
  barrier, width in E_Z ~ Gamma/|d eps/dE_Z|; weaker contact (site240, weak) -> narrower peaks and
  double peaks (split ~2 Gamma/|d eps/dE_Z|) resolved, e.g. site240 lowest peak splits from a = 5 on.

## SCHARF: the constant SOC shift E_so in the onsite energy (2026-10-26, `study_rashba.py` [soc_shift])
* onsite_block adds (alpha^2 + beta^2)/4 (lattice) = E_so = m(alpha^2+beta^2)/2hbar^2, so mu is measured
  from the bottom of the LOWER SOC band and v_F = sqrt(2 mu/m) is alpha-independent.  SCHARF INCLUDES
  THE SAME SHIFT (checked by the user), so the code's mu IS the paper's mu.  Without it mu counts from
  the SOC-free edge and
  v_F = (hbar/m) sqrt(k_mu^2 + k_so^2).  Same alpha in every region => dropping it == mu_S/N/C + E_so.
* Effect (a = 20, L 2 um, tc_barr 0.2, phi pi; E_so = 64 ueV at alpha 16, 255 at alpha 32):
  ABS count, Q region (to one grid step) and Delta_top (66.7/66.5, 94.8/94.1, 33.7/34.3 ueV) unchanged;
  kappa peak POSITIONS move by ~0.05 meV at W 100 (k_F +4.5% -> cos(k_F L) phase +2.4 rad over 2 um),
  heights and the neutral-below / charged-above-~0.9 meV pattern stay; W 300 L 2 fails either way.
  So it is a mu offset, not new physics -- state the convention; only for a paper WITHOUT the shift
  would mu_code = mu_paper + E_so be needed (not Scharf).
* DECISION (2026-10-27): KEEP THE SHIFT (Scharf convention); study_rashba.py [soc_shift] is a side check only.
* Numerical check (`study_rashba.py` [fermi], normal-state electron block of the real lattice blocks, mu_C = 0.7):
  the lattice converges to k_F = k_mu -+ k_so, hbar v_F = hbar^2 k_mu/m = 52.98 meV nm for BOTH
  alpha 16 and 32 (a = 1 nm); the no-shift formula (55.35 / 61.90) is ruled out.  At a = 20 the lattice
  gives hbar v_F = 51.4 (-3%), k_F within 0.2% (alpha 16) / 2% (outer, alpha 32); x and y identical.

## SCHARF: does eps follow Delta_top e^{-L/xi_M} cos(k_osc L)? (2026-10-27/28, `study_oscillation.py`) -- YES
* Context: the kappa ~ 1/2 "plateaus" are two merged eps = Gamma peaks around a maximum of eps(E_Z)
  (det r inside: max +0.16 -> eps_max/Gamma ~ 1.17).  Widest neutral window if
  Gamma ~ (0.72-1) eps_max; parity-crossing dips (eps = 0, det r -> -1) cannot be removed at T = 0.
  Asymmetric Gamma_L != Gamma_R does not help: kappa(E=0) depends only on Gamma_L Gamma_R
  (two-Majorana model scan, symmetric is always optimal at finite T too).  [CONFIRMED in the full model 2026-11-16,
  study_tuning [ep4]: the plateau follows G_L G_R along phi and E_Z.  Strong asymmetry DOES fill the parity-crossing dip
  partly (kappa one maximum 0.47 / 0.42 instead of 1/2 - 0 - 1/2; the far Majorana's tail gives a path at eps = 0) but
  below 1/2, charged and with kappa ~0.05 elsewhere -- no plateau gain.  See the EXCEPTIONAL POINTS section.]
* PREDICTION (cached, scharf_absO_evanescent_W100): complex-k E = 0 solution of the cross-section,
  W 100, a 20, phi pi, E_Z 0.50-1.30 step 0.005 (Q = -1 throughout):
  k_osc = k of the gap minimum (0.0348 -> 0.0439 1/nm), predicted zero spacing pi/(L dk/dE_Z) ~ 0.15 meV;
  xi_M 326 (E_Z 0.5) / 728 (0.8) / 1249 (1.0) / 2418 nm (1.3), next mode ~245 nm (bulk xi);
  hbar v_F/Delta_top agrees only near E_Z 0.9-1.0 (2-2.5x too long at low field).
* MEASUREMENT (cached, scharf_absO_box_W100_L2): eps = lowest |eigenvalue| of the CLOSED box (same
  cross-section, W_S 900 nm, L 2 um), sparse shift-invert, ~5 s/point (161 points in minutes).
  The LDOS route (tools.splitting, transport geometry) costs ~8-9 min/point under 14 workers
  (201 RGF energies to resolve a ueV peak; myf.scan also saves only after the first 12-point chunk).
  Cross-check E_Z 0.8: box 4.09 vs LDOS 4.0 ueV (0.25 ueV grid) -> semi-infinite vs 900 nm SC irrelevant.
* RESULTS (no-parameter tests pass):
  - zeros of eps at E_Z 0.58 0.695 0.83 0.975 1.12 1.27; k_osc L advances by 0.98 0.99 0.98 0.95 0.98 pi
    between them; spacing predicted 0.118 0.137 0.149 0.153 0.152, measured 0.115 0.135 0.145 0.145 0.150 meV.
  - eps_max per lobe 1.6 / 5.3 / 12.2 / 20.5 / 24.3 ueV; ratio to Delta_top e^{-L/xi_M}: A = 1.62 1.66 1.78
    1.73 1.93 (roughly constant, the drift partly from Delta_top's kinks).  Two-parameter fit
    A = 1.83, phi0 = 0.68 pi: rms 0.98 ueV = 4% of max eps.
  - k_osc follows the OUTER Fermi point continuously, also where the gap minimum jumps to k ~ 0 (E_Z > 1.15).
  - hbar v_F/Delta_top (2DEG velocity, global gap) is NOT a usable xi estimate (ratio 0.41 -> 1.5).
  - xi_M SOLVED (cell 1b, dip_point, cached scharf_absO_dip_W100): xi_M is a property of the narrow gap DIP
    at the outer Fermi point k0 alone.  Hyperbola there: xi = hbar v_A / Delta_out with the Andreev-band
    velocity hbar v_A = sqrt(Delta_out E''(k0)) -> xi_hyp/xi_M = 0.86 (E_Z 0.5), 0.96 (0.8), 0.99 (>= 1.0).
    The dip continued to complex k (degree-6 fit of E^2 over k0 +- 2/xi) gives xi_M to < 0.6% everywhere.
    The low-field miss is the lopsided dip: steep outer flank (~70 meV nm) but an almost FLAT inner flank
    at E_Z 0.5, which steepens with field (a hump grows inside the dip).
    hbar v_A = 17.9 -> 68.5 meV nm (0.35 -> 1.33 v_F) while Delta_out = 64.3 -> 28.6 ueV:
    xi_M grows x7.4 from 0.5 to 1.3 meV mostly through v_A (x3.8), less through the gap (x1/0.44).
  - GRID ARTEFACT in tools.gaps: the dip is only ~1/xi wide (0.001-0.003 1/nm), K_GRID steps 0.001 1/nm at
    a = 20 -> the wiggles in Delta_top are misses of the dip minimum, up to ~20% too large (E_Z 1.15:
    43.6 vs 36.0 ueV).  Above ~1.2 the global minimum moves to k ~ 0 and Delta_top < Delta_out is real.
    Any cached gap map (phase_point / gaps) carries this; refine with a 1D minimisation if gaps matter.
    FIXED 2026-11-11: tools.gaps now returns gap_refined's minimum (same grid, no extra grid pass).
    gap_refined vs a brute-force scan (k a step 2e-4 + Brent): identical to 4 digits in ueV at mixed
    a 10 E_Z 0.3/0.5/0.8/0.93/1.15/1.30 (phi pi), 0.93 (1.742 pi) and Rashba a 20 E_Z 1.15 (36.019).
    The plain grid was off by up to 2.4x (mixed a 10, E_Z 1.15: 18.81 vs 7.79 ueV).  STILL plain-grid
    (made before the fix, not rerun): scharf_absR_line_W*, absD_line_*, absS_line_*, absA_xs_a*,
    absO_evanescent_W100 (all pure Rashba).  Mixed caches always used gap_refined.
    dip_point FD step (also fixed 2026-11-11): h = 2e-3 was up to 3.3x the dip width 1/xi_hyp (mixed a 10,
    E_Z 1.35) -> E'' too large, v_A and xi_hyp too LOW: E_Z 1.15 4127 vs 4231 nm (-2.5%), 1.30 9480 vs
    9930 nm (-4.5%); xi_poly only +0.4/+0.7%.  Now h -> 0.1/xi_hyp when coarser (matches h = 3e-5 to
    0.04%).  scharf_absOM_modes_mixed_a10 (cols hv_A, xi_hyp, xi_poly) still holds the old values.
    [REDONE 2026-11-12: study_mixed [oscillation] cell 1 recomputes the dip columns for E_Z >= 0.9 (91 fields) into
    scharf_absOM_dip_mixed_a10_fix and patches them in (the modes cache itself is unchanged; slides_nov.om_modes patches
    too, plots/slides/mixed_lengths.png redrawn).  k0, Delta_out identical; hbar v_A and xi_hyp +0.6 .. +5.7 %, xi_poly
    -0.1 .. -0.7 % (E_Z 1.15: xi_hyp 4127 -> 4230, 1.30: 9480 -> 9929 nm).  Now xi_hyp/xi_M 1.19 (E_Z 0.9) -> 1.40 (1.35),
    1.03-1.40 over E_Z > 0.5; xi_poly/xi_M 1.06-1.19 at E_Z >= 0.55 -- in the MIXED SOC the dip continuation does NOT
    give xi_M to < 1 % as it does for Rashba 16 (not investigated; the band tilt of [angle] is a candidate, untested).]
  - With Delta_out (smooth) in the envelope: A = 1.66 1.68 1.80 1.99 2.01, fit A = 1.92, phi0 0.69 pi,
    rms 0.85 ueV = 3.5%.  A drifts up as L/xi_M -> 1 (asymptotic overlap formula needs L >> xi).
  - THE FIT IS A TEST (2026-11-04, same 2-parameter fit on the L 2 um box, 0.002 grid): right ingredients
    rms 3.4%; naive xi = hbar v_F/Delta_top 18%; xi_M frozen at its E_Z 0.9 value 33%; 2DEG k_F = mu_C/hbar v_F
    20% and no zeros at all; k of the global gap minimum 14% with the zeros misplaced (k0 vs k_osc differ by
    ~0.001 1/nm at low field = 2 rad of k L).  A, phi0 absorb only a constant height and a constant shift.
  - REAL-SPACE CHECK of xi_M (2026-11-04): density of the box Majorana (+-eps pair rotated to its left/right-
    localised combinations, slides_nov.majorana_profiles), slope of ln(density) 0.4-1.6 um from the end, 4-slice
    average: 374 vs 368 nm (E_Z 0.55), 584 vs 580 (0.722), and at 1.05 in a 5 um box 1409-1412 vs 1412 nm.  In the
    2 um box at 1.05 (L/xi_M 1.4) it gives 903 nm: no single-tail region when the box is not >> xi_M.
  - eps_max/Delta_top = 0.02 / 0.09 / 0.24 / 0.43 / 0.70: at L = 2 um the MZM hierarchy eps << Delta_top
    holds only below E_Z ~ 0.85.
  - A vs L (2026-11-03, cell 5, plots/oscillation_A_vs_L; box L = 1.5 / 2 / 2.5 / 3 um): global A 2.03 / 1.93 /
    1.86 / 1.80, phi0 0.68-0.69 pi for all L, rms 4.3 -> 2.5%.  Per-lobe A collapses on L/xi_M for all L:
    1.64-1.70 once L/xi_M >~ 3.5, rising to ~1.8-2.2 as L/xi_M -> 1.  Asymptotic A ~ 1.65-1.70; the 1.92 above
    averaged in lobes outside the formula's range.  (Nilsson et al. Eq. 20, helical edge, hard wall: prefactor
    2 hbar v/xi0 ~ 2 Delta -- different model, same order.)
  - transport (weak probe, a 20) det r -> -1 at 0.562 0.676 0.808: the closed-box zeros shifted by
    -0.02 meV (~0.14 pi, ~ +11 nm effective length ~ half a site): the attached probe moves the end
    boundary.  (Interpretation, not checked.)  NOTE: with tc_barr 0.2 (study_oscillation [fingerprint]) there is NO
    shift (det r minima = box zeros within 0.002 meV) -> the shift belongs to the 'weak' square barrier.

### MIXED SOC at a = 10 (2026-11-04, `study_mixed.py` [oscillation], plots/oscillation_mixed{,_transport}.png)
junction('mixed', 0.7): W 100, L 2 um, t_c 1, mu_C 0.7, phi = pi, E_Z 0.10-1.35 step 0.005; caches
scharf_absOM_*.  Q = -1 on the whole range.  dip_point fixed: near the transition the outer dip is ~a/xi_M wide
and the +-0.15 Brent search missed it (E_Z 1.30 returned the window edge, 71 instead of 2.3 ueV); now also searches
kc +- 0.02; xi_poly = NaN when the poly has no complex zero.
* Delta_top = Delta_out at every field (the global gap minimum IS the outer dip here, unlike Rashba 16).
  Delta_top max 37.5 ueV at E_Z 0.5.  xi_M 284-400 nm below 0.5, then 0.5 / 0.89 / 1.8 / 4.0 / 12.6 um at
  0.6 / 0.8 / 1.0 / 1.2 / 1.35.  xi_hyp/xi_M 0.85-1.3 [1.40 at 1.35 with the 2026-11-12 dip redo].
* THE PHASE HOLDS, THE AMPLITUDE DOES NOT (user caught it on the figure, 2026-11-04).  E_Z 0.45-0.93
  (L/xi_M 1.8-5.3): k_osc L advances 1.00 +- 0.06 pi between box zeros -> zeros/period set by the outer Fermi
  point as for Rashba.  But per-lobe A = eps_max/(Delta_out e^{-L/xi_M}) DRIFTS monotonically with E_Z:
  1.4-2.0 (low field), 1.76 2.27 2.26 2.42 2.77 3.73 5.88 (lobes 0.46 ... 1.155) -- factor ~4; Rashba 16 was
  flat to +-2% for L/xi_M >~ 3.5.  The restricted fit (A 2.71, "rms 2.4% of max eps") hid this: rms relative
  to the LARGEST lobe; per lobe the model is off by up to 35% (0.46 lobe at 0.65 of the model).  The plotted
  global fit (A 4.11) visibly misses the heights.  => Delta_out e^{-L/xi_M} is NOT the mixed-SOC envelope.
  NEVER quote rms/max(eps) for an oscillating quantity spanning decades; check per-lobe ratios.
* WHAT SETS THE AMPLITUDE (2026-11-05, `study_mixed.py` [amplitude], plots/amplitude_mixed.png, cache
  scharf_absAmp_mixed_a10_L2, E_Z step 0.01): the semi-infinite end Majoranas built from ALL 760 complex-k modes
  (hard-wall null vectors, s_min/s_2 <= 3e-11), Rayleigh-Ritz of the box H in span{gamma_L, gamma_R}, NO fit:
  eps_RR/eps_box = 0.9999 (0.9957-1.0002) on all 44 fields with L/xi_M >= 3 -- including the low-field two-mode
  beats; 0.988 (L/xi 2-3), 0.95 (1.5-2), 0.90 (1-1.5), 0.73-1.11 (< 1).  => the lobe heights ARE the overlap of
  the two end Majoranas (same physics as Rashba); what fails is only the single-envelope prefactor A Delta_out.
  - The L-independent prefactor P of the slowest-mode envelope (P e^{-L/xi_M}) drifts against Delta_out:
    P/Delta_out 1.6 (E_Z 0.3) -> 2.0 (0.5) -> 2.5 (0.8) -> 3.2 (1.2).  Up to L/xi ~ 2 the box per-lobe A equals it
    (1.76/1.87, 2.27/2.36, 2.26/2.32, 2.42/2.43, 2.77/2.58 at 0.46 ... 0.87): the drift of A is end-mode structure.
    At L/xi_M < 1.5 the two-end-mode picture itself fails (box above RR by up to 35%) -> per-lobe A 3.7, 5.9.
  - Neither Delta_out nor hbar v_A/xi_M is the right scale: P/(hbar v_A/xi_M) 2.0 -> 2.5 over 0.5-1.3 (spread 1.7x
    including low field).  At L/xi ~ 7 (E_Z 0.2-0.3) P/Delta_out ~ 1.6-1.75 = the Rashba asymptotic value.
  - W (slowest-pair norm / total) = W_R = W_L (left-right symmetric end modes); W 2.8 at E_Z 0.40 where xi_2 ~ xi_M
    (big cancelling weights) -> A_true from the slowest mode alone is meaningless there.
* Low field (E_Z < 0.42): the 2nd complex-k mode has xi_2 257-288 nm ~ xi_M 284-367 nm, weight
  e^{-L/xi_2}/e^{-L/xi_M} 0.14-0.81 -> two-mode beat: zero spacings alternate 0.5 / 1.5 pi, rms 16%.  The slowest
  mode also switches branch (k_osc jumps at 0.2, 0.27, 0.40, 0.49; k_osc differs from k0 by up to 0.002 1/nm).
  Irrelevant for transport: eps < 0.14 ueV there.
* High field (E_Z > 0.93, L/xi_M < 1.8): per-lobe A 3.7 (L/xi 1.1), 5.9 (0.6) -- outside the asymptotic
  formula, as for Rashba but stronger.  Global fit over everything: A 4.11, rms 9.9% (mixes all three regimes).
* TRANSPORT SEES THE BOX eps: Gamma_eff = eps_box/x(det r) on two-Majorana points is smooth while eps spans two
  decades: tau 0.3387 median 1.38 (10-90 % 1.10-1.54) ueV on 49 points; tau 0.5 3.79 (3.06-4.69) on 33;
  p = 2.56 (2.28-2.89) vs study_tuning 2.60-2.71.  Direct Lorentzian Gamma at the eps zeros 0.285 / 0.345:
  1.10 / 1.20 ueV (tau 0.3387, Gamma_eff nearby 1.17), 3.04 / 3.31 (tau 0.5); andreev(0)/2 = 0.498-0.499 there.
  Gamma_eff rises slowly with E_Z (1.1 -> 1.6 ueV over 0.45-0.93), so the constant-Gamma kappa prediction is
  slightly high on the low-field lobes (0.42 vs NEGF 0.35 at 0.54) but tracks every peak position.
  Two-Majorana points end at 0.94: beyond, eps >> Gamma (det r -> +1, excluded by |det r| < 0.95).

## SCHARF: the oscillations as a thermal fingerprint (2026-10-28, `study_oscillation.py` [fingerprint])
* L = 1.5 / 2 / 2.5 um, W 100, a 20, phi pi, tc_barr 0.2, E_Z 0.50-1.30 step 0.002.  Box eps (caches
  scharf_absF_box_L*) + full NEGF at E = 0, T = 0 (scharf_absF_tr_L*; transport_point is 0.6 s/point --
  only the LDOS route is slow) + 0.1 ueV-step NEGF scans around every box zero (scharf_absF_zero_L*_<E_Z>).
* Two-Majorana model holds at T = 0: Gamma = eps_box/x (x from det r) = 2.3 ueV, flat in E_Z up to ~1.0
  and THE SAME FOR ALL L (Gamma is an end/core property) -> drops to 1.6-2 at 1.2.  kappa_model with a
  smooth quadratic Gamma(E_Z) vs NEGF: rms 0.048 / 0.040 / 0.032.
* Zeros: det r minima = box zeros (within the grid); steps of k_osc L/pi between zeros 0.95-1.01 for all L.
  COLLAPSE: zeros at k_osc L/pi = n + c(E_Z), c drifting 0.88 -> 0.79 over the field range but the same
  for every L at a given E_Z (phi0 is an end property, not constant in E_Z).  Spacing ~ 1/L confirmed.
* kappa maxima = 0.499 +- 0.002 in every lobe with 1 < eps_max/Gamma <~ 10 (two per lobe);
  eps_max/Gamma < 1 gives one lower peak exactly on 2x^2/(1+x^2)^2 (0.43 at 0.69, 0.13 at 0.29).
  Lobes with eps_max/Gamma > ~10 overshoot (0.51-0.61) and are charged (|G|/kappa 0.24-0.75).
* DEPTH OF THE ZEROS collapses on L/xi_M for all three L: kappa_min ~ e^{-1.9 L/xi_M}
  (0.10 at L/xi 1.7, 0.023 at 2.5, 0.004 at 3.4-3.7, <1e-4 above 5).  At the minimum G/kappa ~ 0.9-0.94:
  ELECTRON COTUNNELLING through each Majorana's tail reaching the opposite probe.  For L/xi_M <~ 1.1 the
  last box zero (~1.27) has NO kappa zero (det r stays +0.7).
* (see also study_oscillation [branches] below: the outer branch dominates for alpha up to 64 except near the transitions)
* Fingerprint window at tc_barr 0.2: L/xi_M >~ 2.5 (zeros < 0.02), 1 < eps_max/Gamma <~ 5 (neutral 1/2
  peaks, |G|/kappa < 0.07), and kT << Gamma = 2.3 ueV (27 mK) -- the temperature limit of the
  measurability section is unchanged.
* LOBE TOPS vs L (2026-11-04, slides_nov.overlap_L, plots/slides/overlap_vs_L): at the eps maximum of a lobe
  x = c e^{-L/xi_M}, c = A Delta_out/Gamma = 42-47 over all 16 full lobes of L 1.5/2/2.5 -> NEGF kappa there
  follows 2x^2/(1+x^2)^2 to rms 0.004 and peaks (= 1/2) at L/xi_M = ln c = 3.7-3.9.  Beyond that eps < Gamma
  EVERYWHERE in the lobe -> no kappa = 1/2 at all; below it 1/2 survives only on the lobe flanks (narrower,
  charged once L/xi_M <~ 2.5).  At tc_barr 0.2 the length window is 2.5 <~ L/xi_M <~ 3.8 (E_Z 0.8: 1.8-2.8 um);
  a stronger probe moves the upper edge only as ln(1/Gamma).
* kappa(E_Z) FOR L = 2 / 4 / 8 um (2026-11-04, slides_nov.kappa_L, NEGF tc_barr 0.2, a 20, caches
  scharf_absF_tr_L4 / _L8, plots/slides/kappa_vs_L): neutral kappa > 0.45 switches on at E_Z 0.716 / 0.934 / 1.232,
  i.e. where L/xi_M(E_Z) drops to ~3.8 for every L; below it kappa ~ 0 (L 4: < 0.08 up to 0.85; L 8: < 0.1 up to
  1.08).  At the onset eps_max ~ Gamma and the two peaks merge into a flat kappa ~ 1/2 top (L 4: 0.94-0.97, L 8:
  1.27-1.29).  L 4 and 8 stay neutral up to 1.3; L 2 is charged above ~1.1 (L/xi_M < ~1.5).  L 8 only works just
  below the upper transition (~1.35).
* k_osc vs the normal-state band (2026-11-04, Delta = 0, k_y = 0 cut of the junction region, a 20): k_osc =
  0.92-0.93 x the outer Fermi point k_F,out (0.0380 -> 0.0474 1/nm for E_Z 0.5 -> 1.3) and has the same slope
  (0.0091 vs 0.0094 1/nm over that range).  k_F,out solves E_kin(k) - sqrt(E_so(k)^2 + E_Z^2) = 0.636 meV
  (lattice dispersion; E_Z = the sigma coefficient).  The inner branch empties at E_Z ~ 0.64 meV: one Fermi
  point per direction above.  The 7% offset is not explained (strip width / mixing with the S region?).
  [derived 2026-11-10 from this dispersion] dk/dE_Z : dk/dmu_C = E_Z / sqrt(E_so(k)^2 + E_Z^2) (spin canting), ~0.8
  at E_Z ~ 0.93 (E_so ~ 16 x 0.045 ~ 0.72 meV): mu_C and E_Z move k_osc at COMPARABLE rates (E_Z is the sigma
  coefficient, so NOT "E_Z at most half as fast").  CHECKED (full model, slowest complex-k mode, mixed a 10, E_Z 0.93,
  phi 1.742 pi): dk/dE_Z 0.00819, dk/dmu_C 0.01094 1/(nm meV) -> ratio 0.75 (formula 0.80); predicted zero spacing at
  L 2: 192 / 144 ueV, measured on the transport cuts >= 185 / ~139.  k_F,out is a normal-state (Delta = 0) quantity,
  but k_osc is NOT exactly phi-independent: 0.03985 / 0.03926 / 0.03931 / 0.03986 / 0.03993 1/nm at phi 0 / 0.5 /
  1 / 1.5 / 1.742 pi (1.7 % -> k L shifts <= 0.43 pi over the whole phase, ~0.08 meV of E_Z; transport zeros on the
  L 2 grid move 0.85-0.93 with phi), xi_M 695 / 1427 / 1394 / 703 / 669 nm -- at 0.5 and 1 pi another (slower) mode
  looks to take over; 5 phases only, not resolved.  Stronger SOC lowers the ratio -> wider E_Z lobes (alpha 48
  spacing 0.218 vs ~0.145 meV).

* FINITE ENERGY (2026-11-03, cell 6, plots/fingerprint_finite_energy): two-Majorana model (from Nilsson et al.
  PRL 101, 120403, Eqs. 4-6: P_ET = P_CAR = E_M^2 G1 G2 / |Z|^2) -> max_E kappa(E) = 1/2 exactly at E* = sqrt(eps^2 - G^2)
  for every eps >= Gamma.  NEGF, L 2 um, tc_barr 0.2, 52 fields with eps_box > Gamma: kappa(E_peak) = 0.498-0.502
  everywhere, E_peak follows eps_box; kappa(E=0) within 0.03 of 1/2 at only 4% of them.  |G(E_peak)| median 0.02
  but up to ~0.2 at E_Z >~ 1.1 (L/xi_M -> 1).  A linear-response kappa(T) cannot select E* (window ~kT >> Gamma).
## SCHARF: which gap branch sets the MZM tail, outer k0 or k ~ 0? (2026-10-29, `study_oscillation.py` [branches])
* Composite-MF picture (Klinovaja & Loss PRB 86, 085408): the tail mixes an oscillating outer-branch mode
  (Re k = k0) and a non-oscillating k ~ 0 mode; the k ~ 0 mode should win at strong SOC.  Complex k,
  3 slowest distinct modes, W 100, a 20, phi pi, E_Z 0.30-1.80 step 0.01, alpha 16/32/48/64 at fixed mu
  from the band bottom (+E_so convention).  Caches scharf_absB_modes_a<alpha>.
* Upper transition (Q -> +1, k = 0 gap closes): 1.35 / 1.39 / 1.46 / 1.54 meV for alpha 16/32/48/64.
* k ~ 0 mode is the SLOWEST only in narrow windows: alpha 16 never (on this grid), 32: 1.38-1.39,
  48: 1.41-1.46, 64: 1.50-1.54 AND 0.30-0.47 (low field, v at k ~ 0 ~ alpha Delta_ind / E_Z large).
  Everywhere else in Q = -1 the outer branch wins -> oscillating eps, kappa zeros.  The paper's
  strong-SOC statement does not carry over: induced gaps here are small (tens of ueV) and E_Z >> them.
* alpha 48, L 2 um, tc_barr 0.2, E_Z 0.90-1.50 (cell 3; scharf_absB_box_a48_L2, scharf_absB_tr_a48_L2):
  outer-dominated part is a clean fingerprint (box zeros 1.07, 1.288 = det r minima -0.994/-0.963, G ~ 0,
  kappa peaks 0.50, spacing 0.218 meV).  Outer-only model A e^{-L/xi_out}|cos(k L + phi0)| fitted on
  1.00-1.38 (rms 4%) predicts eps ~ 19 ueV through 1.44 and no zero before ~1.51; the box eps instead
  falls 15 -> 5 ueV over the k ~ 0 window 1.40-1.46 (prefactor ~ the closing k = 0 gap: eps -> A Delta_1
  as xi_1 -> inf, so eps FALLS into the transition) and crosses zero EARLY at 1.48 (past the infinite-
  system transition 1.465).  kappa: no zero in the window, monotonic neutral rise 0.09 -> 0.5 (G <= 0.02),
  then the 1.48 dip is shallow and CHARGED (kappa 0.20, G 0.12, det r -0.79).  Window = a quarter lobe.
* Stronger alpha SHORTENS the outer xi_M at fixed E_Z (1.3 meV: 2418 / 1083 / 901 / 982 nm): the
  outer gap grows (spin canting sin theta = alpha k0 / sqrt(E_Z^2 + alpha^2 k0^2)).

## SCHARF: S/N coupling tc and probe coupling tau at a = 10 and 5 (2026-11-02/03, `study_tuning.py` [coupling])
* W 100, L 2 um, alpha 16, mu_C 0.7, mu_S 1, E = 0, T = 0.  DOS (tools.dos_total_phis, exact doubled-system +
  Schur in the phase) for tc 0.5/0.75/0.9/1; kappa, G, det r (tools.transport_phis) for tc 0.75/1 x tau; a = 5 trimmed
  (E_Z step 0.05, 31 phases, tau 0.3387/0.7).  Plots coupling_{dos, kappa_a*, G_a*, 2M_test}; caches scharf_absSN_*.
* Plateau fraction (|kappa-1/2| < 0.03, |G| < 0.1 kappa), a = 10:  tc 0.75: 1.4 / 2.6 / 2.8 / 1.0 %,
  tc 1: 1.6 / 2.9 / 3.0 / 0.4 % for tau 0.2 / 0.3387 / 0.5 / 0.7.  So 0.75 -> 1 barely changes the plateau extent;
  tc 1 is cleaner (neutral share of kappa > 0.25: 89/78/57/12 % vs 74/58/40/6 %) and its kappa lives inside the
  Scharf diamond (V lines, plateaus E_Z 0.35-0.9).  tc 0.75: phi-independent stripes above ~0.7 meV.  Best tau 0.34-0.5.
* a = 5, same bond values: tc 1 (bulk bond, a-independent) reproduces a = 10 -- tau 0.3387: 2.0 % (neutral 93 %),
  tau 0.7: 1.6 %, plateaus in the same E_Z window 0.35-0.9; the same tau is somewhat weaker at a = 5 (map statistics
  suggest the matching tau is ~1.2-1.4x, not the 1D chain's 2x -- NOT a measured transparency).  tc 0.75 collapses
  (0.2 / 1.2 %): the S/N bond is a weaker interface at a = 5 (Q = -1 area 95 % like tc 0.5 at a = 10, DOS V lower).
* Q = -1 fraction of (E_Z <= 2, phi) at mu_C 0.7, a = 10: 98 / 81 / 58 / 49 % for tc 0.5 / 0.75 / 0.9 / 1 (a = 5:
  tc 0.75 95 %, tc 1 48 %).  The Q AREA shrinks with coupling while Delta_top grows ~x7 (study_tuning [topology]).
* DOS: tc 0.9/1 show the gap-closing V exactly on the Q boundary; tc 0.5/0.75 only stripes (gap <= eta 12.5 ueV).

## SCHARF: mu_C and SOC survey for larger plateaus (2026-11-03, `study_tuning.py`, plots/tuning_kappa_tau*)
* a 10, tc 1, W 100, L 2 um, coarse grid (E_Z 0.05, 31 phases).  Plateau fraction (tau 0.3387 / 0.5), Delta_top max:
      SOC          mu_C 0.3              mu_C 0.7              mu_C 1.2
      Rashba 16    2.3 / 3.4 %  81 ueV   2.8 / 4.1 %  67 ueV   0.4 / 1.0 %  30 ueV
      Rashba 32    3.9 / 2.4 %  124      4.5 / 5.4 %  93       0.5 / 0.7 %  42
      Rashba 48    2.0 / 1.7 %  131      5.0 / 6.2 %  100      2.4 / 2.8 %  46
      mixed        2.8 / 3.0 %  48       2.3 / 4.7 %  38       1.0 / 2.2 %  5
  No failed solves.  Best: Rashba 48, mu_C 0.7, tau 0.5 (6.2 %, 85 % of kappa > 0.25 neutral) -- x1.5 over alpha 16,
  not an order of magnitude.  Stronger alpha raises the gap and the neutral share at mu_C 0.7.
* mu_C 0.3: biggest gap, but kappa ~ 0 inside the diamond (only its boundary lines conduct): eps << Gamma there
  (short xi_M); there the WEAKER probe wins (R32: 3.9 % at 0.3387 vs 2.4 % at 0.5).  mu_C 1.2: small gap, charged.
  Mixed SOC: gap ~0.5x Rashba 16, plateau comparable, charged bright patches at high E_Z.
* Reading (untested): the plateau needs eps ~ Gamma over an area; a bigger gap only helps if Gamma is lowered
  with it (eps ~ Delta e^{-L/xi}).  Next: mu_C 0.3 / alpha 32-48 with tau 0.2.  [DONE: a weaker probe does NOT
  enlarge the area -- WEAK PROBE DOES NOT ENLARGE THE AREA below.]
* GAMMA vs PROBE BOND (cell 3): Gamma ~ tau^p.  From x(det r) on two-Majorana points of the two-tau map pairs:
  p = 2.60 (IQR 2.39-2.78, all 12 junctions alike).  Directly (local Andreev Lorentzian at an isolated Majorana,
  R32/R48 mu_C 0.3, E_Z 1, phi pi): Gamma = 4.0 / 4.2 ueV at tau 0.3387 (46 / 48 mK), 11.1 / 12.0 ueV at 0.5
  -> p = 2.63 / 2.71.  Not tau^2.
* WEAK PROBE DOES NOT ENLARGE THE AREA: rescaling x -> x (tau_ref/tau)^p predicted 10-16 % plateau at tau 0.08-0.12
  for mu_C 0.3; NEGF gives 1.0-2.6 % (Gamma 0.1-0.26 ueV = 1-3 mK).  The prediction used points with kappa ~ 1e-4,
  det r ~ -0.999, where the two-Majorana relation holds trivially and 1 + det r is NOT eps/Gamma (relation 0.0013 vs
  NEGF kappa 0.0001).  Rescaling is valid only between maps with sizeable kappa (0.3387 -> 0.5: within 0.002);
  x_of now requires kappa > 0.02.
* THE LENGTH IS THE LEVER (cells 4-5, plots/tuning_kappa_length, tuning_length_tau; R48, mu_C 0.7, coarse grid):
  ln eps spreads by ~ L Delta(1/xi_M) across the diamond, so a shorter L widens the eps ~ Gamma band.
      L [um]   tau 0.3387   0.5           0.55          0.6          0.7     (plateau %, neutral share of kappa > 0.25)
      1.0      4.4 (85)     8.8 (69)      10.9 (56)     8.7 (43)     1.2 (15)
      1.25     --           8.1 (76)      7.0 (65)      --           --
      1.5      6.1 (92)     9.4 (81)      --            --           1.7 (15)
      2.0      5.0 (97)     6.2 (85)      --            --           2.0 (25)
  Optimum L 1-1.5 um, tau 0.5-0.55 (Gamma ~ 11-15 ueV ~ 130-170 mK): ~2.5x the alpha 16, L 2 um baseline (4.1 %).
  L 1.5 / tau 0.5 shows a genuinely extended flat band (E_Z ~ 1.05-1.2 across most of phi).  Shorter L adds charged
  transport (neutral share falls); the predictor (interpolation) overestimated L 1 (13 % vs 10.9 %).
* FINE GRID (cell 6, E_Z 0.025, 61 phases, plots/tuning_best_fine, 2026-11-04): L 1 / tau 0.55: 11.0 % (neutral
  56 %), L 1.5 / tau 0.5: 9.2 % (neutral 80 %); two-Majorana relation on > 1100 neutral points to 0.0004-0.0005;
  no failed solves.  Extended kappa = 1/2 regions with |G| ~ 0: L 1 at E_Z 0.8-1.05 over phi 0.3-1.9 pi, L 1.5 at
  1.05-1.3 over 0.7-1.9 pi.  vs the alpha 16, L 2 um, tau 0.3387 study_tuning [coupling] map (2.9 %): ~3-4x.
  CAVEAT (2026-11-04, from the mixed-SOC check below): the plateau criterion does not exclude the bulk-trivial
  region; the R48 numbers have NOT yet been restricted to Q = -1.
* MIXED SOC = THE THESIS CASE (cell 7, 2026-11-04, plots/tuning_mixed_xi, tuning_mixed_length): Rashba+Dresselhaus
  16 meV nm, theta 0.15 pi, mu_C 0.7, a 10, t_c 1.  xi_M at phi pi (infinite junction): 0.28-0.5 um for E_Z
  0.1-0.6, 0.9 um at 0.8, 1.8 at 1.0, 7.2 at 1.3 (Delta_top max 37.5 ueV at 0.5); mu_C 0.3: 0.34-0.5 um to E_Z 1.
  Plateau % (tau 0.3387 / 0.5 / 0.7), all points // only Q = -1 points:
      L 1.0:  2.3 / 2.0 / 1.9   //  2.1 / 1.5 / 1.5
      L 1.5:  1.5 / 5.3 / 1.7   //  1.4 / 3.2 / 1.1
      L 2.0:  2.3 / 4.7 / 0.9   //  2.0 / 2.7 / 0.6
      L 3.0:  2.8 / 3.9 / 0.3   //  1.8 / 2.7 / 0.2
  The length lever does NOT carry over: ~2-3 % in Q = -1 everywhere.  The two-Majorana relation
  |kappa - (1 - det r^2)/2| on neutral kappa > 0.3 points: 0.0002-0.0013 inside Q = -1 for every L, tau, but
  0.03-0.16 in the bulk-trivial region E_Z > 1.3 (neutral by |G| < 0.1 kappa, NOT two-Majorana) -- 27-43 % of the
  tau 0.5 "plateau" points are there.  => plateau criterion must also require Q = -1 (or the 2M relation).

## SCHARF: the best kappa = 1/2 plateau of the MIXED SOC, as a line (2026-11-04/05, `study_mixed.py` [plateau])
plots/plateau_mixed_best.png; caches scharf_absPlat_*.  Plateau point = |kappa-1/2| < 0.03, |G| < 0.1 kappa, Q = -1
(Pfaffians on every phase) and kappa = (1-det r^2)/2 within 0.01.  Coarse maps: runs along E_Z at fixed phi are
<= 0.15 meV, along phi up to ~1 pi -> search along phi (one RGF set-up per field, phases nearly free; 241 phases).
* Fine grids (E_Z +- 0.1, step 0.01) around the 4 best coarse runs, then tau x0.8/x1.25 and L +-0.5 um around the
  winner, then E_Z refined to 0.0025.  WINNER: mu_C 0.7, tau 0.5, L 2 um, E_Z 0.930 meV (B ~ 3.2 T at g 10):
  plateau over 1.37 pi of phase (phi 1.06 pi -> 2 pi -> 0.42 pi), kappa 0.476-0.4998 (mean 0.491), |G| <= 0.047,
  det r -0.19..+0.22 (x = eps/Gamma within ~0.82-1.22), 2M residual <= 6e-4, Q = -1 on every phase.
  Runner-ups: L 2.5 tau 0.5 E_Z 1.02 (1.22 pi); tau 0.4 L 2.5 E_Z 1.00 (1.02 pi); mu_C 0.3 tau 0.3387 L 2 E_Z 1.25
  (0.97 pi, Q = -1 checked).  Sensitive in field: 1.23 / 1.37 / 1.15 pi at 0.9225 / 0.930 / 0.935.
* Along E_Z at the plateau centre (phi 1.74 pi): plateau only 0.925-0.970 meV = 50 ueV (~0.17 T).  It is the
  eps = Gamma maximum of kappa = 2x^2/(1+x^2)^2 held over a phase range (det r ~ 0 throughout), NOT a protected
  quantisation: kappa <= 1/2 on two-Majorana points, = 1/2 only at x = 1.  E = 0 (T -> 0); Gamma ~ 3.8 ueV ~ 44 mK
  sets the temperature scale (finite T not computed).  [CORRECTED 2026-11-24 from the [ep8] poles (pole = the Gamma
  definition, Capecelatro et al.): at the winner Gamma_1 (one probe) = 8.36 ueV, pair poles +-6.80 - 8.36i ueV; the 3.8
  ueV was the phi = pi Gamma_eff, not this point.  k_B T_half ~ 0.35 min|lambda| = 3.77 ueV = 44 mK -- the same 44 mK by
  coincidence, and the computed kappa(T) agrees ([nh]: 0.496 / 0.486 / 0.411 / 0.189 at 0 / 10 / 20 / 50 mK).]
* phi = pi at 0.930 is NOT part of it: kappa 0.469 but G 0.107 (0.23 kappa), box eps 0.41 ueV vs det-r x 0.76 ->
  not two-Majorana (box zero 0.93; the NEGF det r minimum sits ~0.01 meV lower).
* L 3 um, tau 0.5, phi = pi (2026-11-10, [plateau] cell 5, plots/plateau_mixed_L3.png): the coarse column (step 0.05)
  looked like the best plateau yet -- kappa 0.473 / 0.499 / 0.472 at 0.85 / 0.90 / 0.95, neutral, det r monotonic.
  ALIASING: the fine cut (step 0.0025) has eps zeros at ~0.877 and ~0.97 (kappa 0.02 / 0.07, det r -> -1) between
  the coarse points, which sat on three SEPARATE eps = Gamma windows.  Plateau windows 23 / 12 / 10 / 5 ueV (vs 50
  at the L 2 winner); phase extent <= 0.24 pi (0.99 meV) vs 1.37 pi.  Longer L = denser oscillation in field
  (zero spacing ~0.09 meV here) = narrower windows.  Never read a plateau width off the 0.05 coarse maps.
* FIELD vs GATE (2026-11-10, [plateau] cell 6, plots/plateau_mixed_muC.png; only the strip mu_C moves, probes keep
  mu_N -- replaces the old pre-fix mu scan).  Plateau windows (neutral, 2M, Q = -1) through the point:
      E_Z at phi 1.742 pi (the winner)       50 ueV (0.17 T)   kappa >= 0.4: 85 ueV   lobe top x_max 0.88 (det r -0.13)
      mu_C at phi 1.742 pi, E_Z 0.930        35 ueV            kappa >= 0.4: 62 ueV   same lobe top
      E_Z at phi 1.425 pi (0.9113-1.0012)    90 ueV (0.31 T)   kappa >= 0.4: 110 ueV  x_max 1.2 (det r +0.19), two
                                                                                       merged eps = Gamma peaks
  The 1.425 pi line is the widest run ALONG E_Z on the stage-1 grid (also at 0.08-0.11 pi); it is the notes'
  optimum Gamma ~ (0.72-1) eps_max.  The phase-run winner and the field-run winner are different points.
  Gate windows ~0.7 x field windows (k_osc moves 1/0.75 x faster in mu_C).  No plateau beyond one lobe top on any axis.

## SCHARF: kappa peaks and the sign of det r -- bulk transition or eps = Gamma? (2026-11-04, study_tuning cell 8)
* Mixed SOC, mu_C 0.7, L 1/1.5/2/3 um x tau 0.3387/0.5/0.7 coarse maps; bulk Q on the same grid
  (scharf_absTune_Q_mixed_mu0.7, 45 % of the map Q = -1).  Plots detr_kappa_mixed (kappa, white det r = 0,
  red dashed Q boundary), detr_crossings_mixed.
* det r = 0 => r singular => one fully transmitted eigenchannel => kappa >= 1/2.  27 det r = 0 crossings refined by
  regula falsi (5 NEGF solves each, |det r| <= 2e-2): neutral ones kappa 0.493-0.5005 (|G| < 0.07), charged ones
  0.50-0.94.  (The 0.493-0.496 values sit at E_Z <= 0.03, phi pi: the gap-closing tip; eta absorption -- CHECKED
  2026-11-11: kappa 0.4894 / 0.4960 at eta x1 / x0.1 there, limit ~0.497.)  One non-converged (L 3, E_Z 0.01, |det r| 0.66) excluded.  61-83 % of the kappa peaks > 0.3 sit within
  one grid step of a det r sign change; the rest are charged peaks where det r does not cross.
* But only 23-36 % of the det r sign-change links lie on the bulk Q boundary (<= 1 grid step).  The discriminator
  is tau: links ON the Q boundary stay on the same link from tau 0.3387 to 0.5 / 0.7 in 82-93 / 72-82 % of cases,
  INTERIOR links only 36-45 / 9-13 %.  So the boundary ones are properties of the junction (bulk transitions,
  mostly the lower V where Delta_top opens fast), the interior ones are the eps = Gamma crossover (move with Gamma).
  det r also changes sign in the TRIVIAL region (E_Z ~ 2, charged, kappa 0.6-0.94): no transition at all there.
* phi = pi column: det r ~ -1 (kappa ~ 0) at low E_Z for L >= 1.5, then turns positive INSIDE Q = -1 where xi_M
  grows (L 1.5: ~0.6, L 2: ~0.7, L 3: ~0.8 meV) -- far below the bulk transition at 1.35, where det r ~ +0.95 on
  both sides (xi_M >> L: the finite junction cannot see it).  The crossing moves towards the bulk boundary as L
  grows, as Akhmerov's large-L limit requires; Q-boundary links with a det r crossing nearby: 60 -> 70 %
  (tau 0.3387) and 76 -> 94 % (tau 0.7) from L 1 to 3 um.
* WHY the det r = 0 lines look smoother at strong tau (2026-11-10, cache read only, inside Q = -1).  Two-Majorana:
  det r goes -1 -> +1 while eps goes ~Gamma/3 -> 3 Gamma, so the det r < 0 strip around an eps zero is ~2 Gamma /
  |grad eps| wide -- at weak tau narrower than the grid step (E_Z 0.05, phi 2pi/30).  Median |det r| jump across a
  sign change 1.08-1.30 (tau 0.3387) vs 0.70-0.86 (tau 0.7); links resolved (jump < 1) 28-44 % vs 72-88 %; det r < 0
  pieces 6-16 vs 2-5.  BUT neutral links (|G| < 0.1 kappa both ends) are 57-69 % / 30-40 % / 0-1 % at tau
  0.3387 / 0.5 / 0.7: the smooth tau 0.7 lines are CHARGED, not eps = Gamma (det r = 0 only means one channel fully
  transmitted).  Untested reason: Gamma no longer << Delta_top, so the probe transmits through gapped states too.

## SCHARF: "resonance, not a topological plateau" -- advisor figures, mixed SOC (2026-11-06, slides_nov.py)
* collapse_mixed (plots/slides/mixed_collapse.png; caches scharf_absOM_*, L 2 um, a 10, phi pi): NEGF kappa vs
  x = eps_box/Gamma with Gamma MEASURED (local Andreev Lorentzian, 1.15 / 3.18 ueV at tau 0.3387 / 0.5), no fit.
  E_Z <= 0.94 (separated pair): |kappa - 2x^2/(1+x^2)^2| median 0.013 / 0.006, 90 % 0.085 / 0.112; max kappa 0.501 /
  0.499; the visible outliers are the charged points (7 / 19 of 85, |G| > 0.1 kappa) and the slow Gamma drift.  The
  two taus put their kappa = 1/2 points at different fields but on one curve in x.
  [REDONE 2026-11-24 with the pole Gamma at every field (study_tuning [gamma]): collapse_mixed / mixed_overlap /
  mixed_oscillation now use slides_nov.pole_line (x = eps_eff / Gamma, both from the poles); median 0.000 / 0.001,
  90 % 0.006 / 0.020 -- the 'slow Gamma drift' scatter is gone.  See "GAMMA ALONG phi = pi" under [regimes].]
* length_detr_mixed (plots/slides/mixed_length_detr.png; study_tuning cell 7 coarse maps, tau 0.5, phi = pi column):
  det r < -0.9 with kappa <= 0.07 up to E_Z 0.40 / 0.50 / 0.55 / 0.80 meV for L 1 / 1.5 / 2 / 3 um; first neutral
  kappa > 0.3 at 0.15 / 0.55 / 0.70 / 0.85.  Better-separated Majoranas kill kappa and pin det r = -1 (= Q) -- the
  opposite of a topological plateau.  eps = 0 can only be approached exponentially (e^{-L/xi_M}) and gives kappa = 0.

## SCHARF: is kappa = 1/2 really eps = Gamma? Parameter-free test (2026-11-03, study_tuning [coupling] cell 2)
* Two-Majorana model: det r = (x^2-1)/(x^2+1), kappa = 2x^2/(1+x^2)^2 (x = eps/sqrt(Gamma_L Gamma_R))  =>
  kappa = (1 - det r^2)/2, two full-model outputs (transmission vs left-lead reflection), no fit.  kappa = 1/2
  <=> det r = 0 <=> x = 1, and neutral kappa = 1/2 means R_ee = R_he = T_ee = T_he = 1/4.
* Only meaningful at LARGE kappa: weak normal transmission gives (1 - det r^2)/2 = kappa - kappa^2/2 anyway.
  A plain electron channel with kappa = 1/2 would have det r = 1/2 and miss by +0.125.
* a = 10 coupling maps (all phi, all E_Z), kappa > 0.3:
      map (tc, tau)      neutral (|G|<0.1 kappa): n, |res| median / 95%     charged: n, res median
      0.75, 0.2          177   0.0070 / 0.012                               66   +0.160
      0.75, 0.3387       297   0.0018 / 0.004                              221   +0.039
      0.75, 0.5          400   0.0005 / 0.0015                             640   +0.066
      0.75, 0.7          155   0.0004 / 0.0010                            2259   +0.065
      1.0,  0.2          316   0.0037 / 0.010                               39   +0.004
  Neutral large-kappa points ARE the two-Majorana form (the neutral 1/2 points have |det r| <= 0.24, i.e.
  eps/Gamma 0.78-1.28).  Charged ones carry extra transmission (kappa above the prediction, up to ~0.4) and can
  cross 1/2 without eps = Gamma -- the majority at tau 0.7.  So the |G| < 0.1 kappa plateau criterion is what
  selects the eps = Gamma mechanism.  Not shown: that x matches eps_box / Gamma independently (box splitting).

## SCHARF: is the topological regime reachable at the coupling-study parameters? (2026-11-02, `study_tuning.py` [topology])
* Cross-section only (W 100, a 10, alpha 16, Delta 0.25, phi pi, W_S 900): Q, Delta_top (tools.gap_refined:
  banded eigensolver, grid minimum refined in k -- the old ~20% grid overestimate is 1-4% here) and xi_M
  (study_oscillation.slowest_mode, exact).  Grids: mu_C 0-2.4 x E_Z 0-2 (step 0.1) at mu_S 1; mu_S 0.5-3 x E_Z
  at mu_C 0.7.  Caches scharf_absTopo_*; plots topology_muC / muS / xi.
* At the working point mu_C 0.7, mu_S 1 (L = 2 um):
      tc     Q=-1 at E_Z    max Delta_top   xi_M min..max     max L/xi_M
      0.5    0.1-2.0        9.2 ueV         4.7 - 49 um       0.43   -> the two ends overlap fully
      0.75   0.1-1.7        28.5 ueV        1.3 - 10 um       1.52
      1.0    0.1-1.4        66.5 ueV        0.26 - 2.8 um     7.8 (low field, E_Z 0.1-0.5)
  tc 1 reproduces the a = 20 Delta_top 66.7 ueV (bulk bond, a-independent).
* Over mu_C: Q = -1 covers 55 / 57 / 70 % of the map (tc 0.5 / 0.75 / 1); best gap 9.7 / 31.3 / 83.9 ueV, all
  at mu_C ~ 0-0.3; best L/xi_M 0.39 / 1.44 / 6.55.  xi_M is flat below mu_C ~ 0.8-1.0 and grows steeply above.
  mu_C 0.7 is close to optimal; mu_C >~ 1.2 shrinks both region and gap.
* Over mu_S (mu_C 0.7): best gap 12.9 / 38.2 / 88.7 ueV at mu_S 2.5 / 2.25 / 1.75 (vs 9.2 / 28.5 / 66.5 at mu_S 1):
  +30-40%, but at other fields (fixed E_Z 0.6, tc 0.75: mu_S 2 gives 17.4 vs 27.8 ueV).
* So the S/N coupling mostly buys GAP and SHORTER xi_M, not area: the area grows only 55 -> 70 %, the gap x7,
  xi_M /10.  At L = 2 um only tc ~ 1 gives separated end modes (L/xi_M > 3); tc 0.75 needs L ~ 4-7 um,
  tc 0.5 L >~ 15 um (from L/xi_M = 3-5; NOT checked in transport).  tc 0.5's gap is below the DOS eta 12.5 ueV.
  Not done: a = 5 (the same tc is a different interface there).

## SCHARF: (E_Z, phi) maps of kappa and total DOS at a = 5 (2026-11-10, `study_tuning.py` [maps])
* Mixed SOC, mu_C 0.7, tc 1, W 100, L 2 um, a 5; probe bond tau 0.2 / 0.5 / 1; E_Z 0-2 step 0.025 (81) x phi 121;
  eta 1e-4, LINEAR colour scales.  Total DOS(E = 0) = -Im Tr G / pi WITH the probes attached (tools.dos_total_phis
  tau=).  Q (Pfaffian) on the half grid (E_Z step 0.05 x 31 phases) -- the dashed line is stair-stepped by that.
  Caches scharf_absMap_{tr,dos}_mixed_a5_tau*, scharf_absMap_Q_mixed_a5; plots maps_mixed_a5_{kappa,dos}_tau*.png.
* Q = -1 on 45.6 % of the map.  Boundary: a V from phi = pi, E_Z ~ 0.025 to phi = 0 at 0.575, and a nearly FLAT
  top at 1.225 (phi 0) .. 1.375 (phi 0.75-1.25 pi) -- only the LOWER half of the Pientka diamond (PRX 7, 021032);
  Q = -1 at every phase for E_Z 0.6-1.2.  The bright gap-closing lines of the DOS (clearest at tau 1) lie ON the Q
  boundary: the edges are sharp k_x = 0 closings, not smeared.  Own reading, UNTESTED: mu_C 0.7 is not >> E_Z and the
  inner spin band empties near 0.64 (pure Rashba a = 20 value), so above it no two-band compensation condition ->
  phi-independent.  Test: the Q map at mu_C ~ 1.5-2 should recover more of the upper half.
      tau   kappa max   kappa > 0.25   neutral of those   plateau (also Q = -1)   DOS min .. max [1/meV]
      0.2   0.782         5.9 %         0.98              1.1 % (1.1 %)          0.052 .. 2320
      0.5   0.971        13.5 %         0.88              2.7 % (2.4 %)          0.129 .. 1630
      1     1.896        45.6 %         0.01              0.1 % (0.0 %)          3.57 .. 1060
  (tau 1 max: ONE pixel, the gap closing at E_Z 0, phi pi, G = kappa; 1.40 without it.  0 failed solves anywhere.)
* Inside Q = -1: bright DOS plateau = 2/(pi Gamma) (eps << Gamma, kappa ~ 0); the fan of rays / horizontal stripes
  with kappa ~ 1/2 and isolated DOS spikes = the oscillating eps crossing Gamma / zero (study_oscillation) -- NOT phase
  boundaries.  Dark above ~0.6 where xi_M grows.  At tau 1 kappa ~ 1 is CHARGED (neutral 0.01): a tongue from
  (pi, 0) via phi ~ 0.75 pi up into the trivial region.  NOT worked out: why the rays run diagonally and mainly at
  phi < pi; the curved + vertical lines and the ring pattern ABOVE the top boundary (trivial, 1.3-2 meV).
* Running it: ~11 min per kappa field / ~7 min per DOS field on a one-thread worker (121 phases), up to ~24 min when the
  laptop throttles (95 C, ~1 GHz under 8 workers -- nothing to do in software); Q 44 s per point under load.
  scan / run_scan take chunksize=: the default (one chunk per worker) returns NOTHING until the chunks end, so a killed
  run loses everything; chunksize=1 saves every point and resumes.  Background tasks die with the Claude session, so
  long runs were launched detached (WMI Win32_Process Create with Win32_ProcessStartup ShowWindow = 0 -- with a
  visible console, closing that window kills the run; it did twice).

## SCHARF: lead solver and eta, checked on the 2026-11-09/10 runs (2026-11-11, scratch checks, no study part)
* Caches of the a = 5 maps (kappa, DOS, Q) and every [plateau] cut / grid of 11-10: 0 failed lead solves, 0 NaN; no
  overflow / self-consistency warnings in any log (worker stderr goes to the logs).
* WRONG-BRANCH TEST (the failure N_UNRELIABLE cannot see, Gresta eta 1e-7 below): production surface onsite vs the
  eigenmode solution from the M smallest-|lambda| Bloch modes (explicitly decaying).  Probe leads (open channels at
  E = 0): agree to 2e-13 (a 5, eta 6e-8 t and the DOS eta 2.5e-6 t) and 1e-15 (a 10); raw Sancho-Rubio residual
  9e-14 / 6e-16 (no Newton needed); |lambda| kept <= 1 - 1.35e-7, dropped >= 1 + 1.35e-7 at the smallest eta -- the
  branch choice is unambiguous.  SC ribbons (gapped at E = 0, all a, L, phi, eta): residual ~1e-16.  Gamma >= 0 and
  Im Tr g < 0 (retarded) everywhere.  The top ribbon at phi is a gauge rotation of phi = 0, so two phases suffice.
* ETA IN TRANSPORT (eta_rel x 0.1 / 1 / 10): kappa changes linearly ((x10-x1)/(x1-x0.1) = 9.3-10.0, no jumps).
  Production eta lowers kappa by 2.5e-4 (L 2 field run), 3.1e-4 (L 2 winner), 5.9e-4 (L 3, pi), 7.7e-4 (a 5 tau 0.5)
  -> plateau numbers stand.  Gap-closing tip (E_Z 0.02, pi): 6.6e-3 (0.4894 vs 0.4960 at eta/10).
  det r NEAR -1 IS ETA-LIMITED at weak coupling: a 5, tau 0.2 (Gamma ~0.35 ueV), E_Z 0.3, 1.2 pi: det r -0.9665 at
  production eta, -0.9965 at eta/10 (kappa 7e-5): 1 + det r ~ a few eta/Gamma there (absorption), not eps/Gamma --
  the reason x_of needs kappa > 0.02.
* ETA IN THE DOS MAPS (eta 1e-4 meV): a 5, tau 0.2, E_Z 0.3 -- V plateau (1.2 pi) 1691 / 1075 / 233 per meV at eta
  1e-5 / 1e-4 / 1e-3 meV = 2/(pi (Gamma + 2.4 eta)) with Gamma 0.35 ueV to ~1 % -> the tau 0.2 map plateau is ~40 %
  below its eta -> 0 value (~1810); tau 0.5 ~6 %, tau 1 ~1 % (same formula, not recomputed).  V-edge gap-closing line
  2507 / 1304 / 230: its brightness is largely eta.  Background (0.1, 0.2 pi) 0.015 / 0.061 / 0.52 ~ eta (tails).
  Shapes unaffected; quote eta with any DOS value, or rerun tau 0.2 at 1e-5 (~3 h).

## SCHARF: the MIXED SOC vs BOTH lead couplings, tc and tau (2026-11-11, `study_tuning.py` [leads], plots/leads_mixed.png)
* Thesis junction (mixed_soc(16, 0.15 pi), W 100, L 2 um, mu_C 0.7, mu_S 1, a 10).  tc = S/N bond (tc_top = tc_bot;
  1 = bulk bond; 1.25 = STRONGER than a bulk bond, MODEL EXTRAPOLATION).  tau = probe bond: enters only Gamma, not
  Q / Delta_top / xi_M.  phi = pi bulk line (E_Z step 0.05, refined gap), xi_M at E_Z 0.2-1.2 (slowest_mode), coarse
  Q map (21 x 16), transport on the [tuning] coarse grid, phi = pi lines step 0.01.  Caches scharf_absLeads_*; reuses
  tr_name(...) maps and scharf_absOM_line_mixed_tau*.  0 failed solves.  ~2 h detached (shared CPU); redraw from cache
  `python study_tuning.py leads` in ~1 min.  The pure-Rashba counterpart is [coupling] / [topology] above.
* SC LEADS (phi = pi; transport at tau 0.5):
      tc     Q=-1 area   Delta_top max (E_Z)   xi_M min (E_Z)    max L/xi_M   plateau in Q=-1   neutral share
      0.5    95 %        0 -- GAPLESS          undefined         --           0.5 %             0.14
      0.75   77 %        9.8 ueV (0.60)        2.36 um (0.4)     0.85         4.6 %             0.44
      0.9    53 %        24.3 (0.55)           0.92 um (0.4)     2.17         4.2 %             0.64
      1      45 %        37.5 (0.50)           0.28 um (0.2)     7.03         2.6 %             0.83
      1.25   51 %        8.1 (1.45)            4.7 um (1.2)      0.42         1.1 %             0.35
  tc 1 reproduces the numbers above (Delta_top 37.5 at 0.5, xi_M 284 / 298 / 496 / 885 / 1805 / 4047 nm).
  tc 1.25: Q = -1 at phi pi only from E_Z 0.9, Delta_top 0.7-8 ueV.
* tc 0.5 IS GAPLESS at every field with Q = -1 (refined Delta_top <= 1.6e-4 ueV; a band changes sign at finite k,
  checked at E_Z 0.6 (k 0.0262/nm) and 1.2 (0.0355/nm)): Q = -1 without a gap, so no protection and no xi_M.  The same
  holds for tc 0.75 at E_Z >= 1.2.  Pure Rashba keeps 9.2 ueV at tc 0.5: the mixed SOC needs tc >~ 0.75 for a gap at
  all.  TRAP: slowest_mode drops the |lambda| = 1 (propagating) modes and returns the 2nd complex-k mode (257 nm,
  k 0.0404/nm, field-independent; cf. xi_2 in the MIXED SOC subsection).  It is masked via the bulk line
  (CLOSED_L = 1e-5 meV).
* tc -> PROTECTION: the bulk bond is the optimum.  tc 0.5 -> 1 takes Delta_top 0 -> 37.5 ueV and min xi_M inf -> 0.28 um,
  the largest lever in the model.  Beyond a bulk bond protection is lost again (tc 1.1: Q = -1 at pi from E_Z 0.40,
  max 29.7 ueV; tc 1.25: from 0.90, max 8.1).  WHY: tested, see "CHECKS (cell 5)" below -- the strong bonds reflect.
* tc -> PLATEAU: the largest fraction is at tc 0.75-0.9 (4.6 / 4.2 %), but those maps are mostly charged (neutral
  0.44 / 0.64): xi_M ~ L, so eps is large and oscillating and crosses Gamma more often.  tc 1 has fewer crossings but
  cleaner ones (0.83).  The plateau fraction counts eps = Gamma crossings, not protection; it falls where protection
  rises.  => The advisor's "tc helps protection" HOLDS for the mixed SOC (up to the bulk bond); it does not help kappa.
* PROBE LEADS (tc 1).  Gamma = local-Andreev half width at the eps zeros E_Z 0.285 / 0.345, phi pi (tau 0.3387 / 0.5
  equal the MIXED SOC subsection's direct-Lorentzian values):
      tau      Gamma [ueV]   plateau in Q=-1   neutral share   phi = pi line: kappa = 1/2 windows, widest (E_Z)
      0.2      0.35          0.9 %             0.93            2, 0.01 meV
      0.3387   1.15          2.0 %             0.90            3, 0.02
      0.5      3.2           2.6 %             0.83            2, 0.06
      0.7      10.0          0.5 %             0.14            1, 0.02
      1        37.4          0.1 %             0.01            0
  [2026-11-24: these are local-Andreev HWHMs; = the exact pole width to <= 1.6 % for tau <= 0.7, but at tau 1 the pole is
  52.1 ueV (the line overlaps bulk poles) -- see "GAMMA THREE WAYS" under [regimes].  Gamma = exact pole from now on.]
* Gamma(tau) is NOT one power law: local d ln Gamma / d ln tau = 2.2 / 2.6 / 3.4 / 3.7 between neighbouring tau
  (global fit 2.88 over 0.2-1; over 0.2-0.7 it is 2.64, matching the earlier p 2.56-2.71).  The weak-coupling end follows the golden-rule tau^2.
  Why it steepens: not worked out.
* tau -> PLATEAU: the optimum is tau 0.34-0.5 (Gamma 1-3 ueV).  kappa = 1/2 needs eps = Gamma.  The window in E_Z grows
  with Gamma (~ Gamma / |d eps/dE_Z|) as long as Gamma stays below the eps envelope.  tau 1 (Gamma 37 ueV >= Delta_top
  at every field except E_Z 0.50-0.51): eps is a subgap level, so eps < Delta_top <= Gamma and no eps = Gamma crossing
  exists (0 windows).  kappa is CHARGED at every field; the earlier "charge starts where Delta_top < Gamma" was WRONG,
  see CHECKS.  tau 0.7: neutral share 0.14.
* CHECKS (2026-11-12, cell 5 of [leads], plots/leads_checks_mixed.png; caches scharf_absLeads_{bulk,Q}_tc{0.8,1.1}).
  WHY tc > 1 LOSES PROTECTION.  Control tc 0.8: on the lattice ONE bond x transmits exactly as 1/x (T(x) = T(1/x);
  normal state, E_F, k 0: 0.657 vs 0.659 for 1.25), yet tc 0.8 is an ordinary weak contact (Q = -1 at pi from E_Z 0.15,
  Delta_top 14.1 ueV, Q=-1 area 70 %).  So the barrier strength of one bond is NOT the cause.  Cross-section, k 0:
      tc     T one bond   T junction   bound levels [meV] (within 100 nm)   ABS(pi, E_Z 0)   Q=-1 at pi from   max Delta_top
      0.8    0.657        0.707        none                                 65.6 ueV         0.15              14.1 (0.55)
      1      1.000        0.992        none                                  2.8             0.05              37.5 (0.50)
      1.1    0.915        0.654        -1.034 (77 %)                        92.4             0.40              29.7 (0.70)
      1.25   0.659        0.209        -1.44 (99 %), -1.26 (97 %)          189.4             0.90               8.1 (1.45)
  (T per spin; T junction = S | tc | N strip | tc | S; bulk S band bottom -1.000 meV at k 0; bound levels first at
  tc 1.096 on a 3 % grid, never for tc <= 1.)  Two strong bonds turn the strip into an attractive well: it binds levels
  below the S band bottom AND reflects the k = 0 channel at E_F (junction T 0.21 at 1.25 while one bond still passes
  0.66).  Two weak bonds form a cavity instead (Fabry-Perot: T = 1 and ABS(pi) = 0 near tc 0.6).  The reflection opens
  the phi = pi ABS gap, which E_Z must close before Q = -1: onset 0.05 -> 0.40 -> 0.90 for tc 1 -> 1.1 -> 1.25.  Past the
  late transition the gap reopens only to 8 ueV: at E_Z 1.0-1.45 the gap minimum is an extended state at small k
  (0.013-0.029 /nm); from 1.55 a junction-bound state (k 0.052 /nm, 100 % within 100 nm).  Numbers computed; tying the
  bound levels and the reflection to one "well" is the interpretation.  => tc 1 is optimal because the bulk bond is the
  only reflectionless bond on the lattice; "stronger than bulk" is not a better contact here (and is a MODEL EXTRAPOLATION).
  THE tau 1 CHARGE IS NOT SET BY Delta_top < Gamma.  (i) tau 1 is charged at EVERY field: T_he / T_ee = 0.01-0.16 for
  E_Z 0.2-1.2 (up to 0.77 at 1.29, next to the gap closing).  T_ee is smooth and grows roughly like exp(-2L/xi_M) (ratio
  1.2-25 over six decades); T_he carries the eps oscillation.  The "onset at 0.8" was only kappa passing the 0.25
  threshold of the neutral-share statistic.  (ii) Median G/kappa (kappa > 0.05) per 0.1 meV window is the SAME for
  tau 0.2 / 0.34 / 0.5: 0.00 below E_Z 0.7, 0.03-0.05 at 0.8-0.9, 0.09-0.13 at 0.9-1.0, 0.23-0.26 at 1.0-1.1, 0.42-0.46
  at 1.2-1.3, although Delta_top / Gamma spans 1-100 there: at weak coupling the charge is a junction property that grows
  as L/xi_M falls (3.5 -> 0.5).  tau 0.7 lies 1.5-3x above that at every field, with no step at E_Z 1.10 where
  Delta_top crosses its Gamma.  Reading (not derived): at weak coupling kappa and G both scale ~ Gamma^2 (median G over
  E_Z 0.95-1.10 grows x9.3 / x7.2 where Gamma^2 grows x10 / x7.6), so G/kappa is fixed by the junction; once Gamma
  passes eps the neutral Majorana part saturates while the smooth electron cotunnelling keeps growing.

## SCHARF: the advisor-deck runs in the MIXED SOC (2026-11-11/12, `study_mixed.py` [deck], slides_nov mixed_*)
* Caches scharf_absDeck_*mixed* (all complete), 6 workers (8 crashed the laptop).  Figures plots/slides/
  mixed_kappa_vs_L, mixed_SN_coupling, mixed_temperature, mixed_best_temperature (redraw: slides_nov.py mixed_kappa_L
  mixed_sn mixed_temperature mixed_best_T, ~1 min from cache).
* LENGTH (a 10, tau 0.3387, phi pi, E_Z step 0.002; L 2 / 3 / 4 um): neutral kappa > 0.45 from E_Z 0.602 / 0.740 / 0.824
  = L/xi_M 4.01 / 4.11 / 4.17 (the Rashba 3.8 rule: 0.625 / 0.765 / 0.855); first charged peak (|G| > 0.1 kappa,
  kappa > 0.3) at 0.918 / 0.976 / 1.072 = L/xi_M 1.5 / 1.8 / 1.7; best neutral kappa 0.498 at every L.
  => mixed window ~1.7 <~ L/xi_M <~ 4.1 (Rashba 2.5-3.8).
* S/N (topo_point on (mu_C 0-2.4, E_Z 0-2), tc 0.5 / 0.75 / 1): Q = -1 area 48.4 / 50.3 / 68.0 %, GAPPED (Delta_top >=
  0.01 ueV) only 0 / 24.0 / 29.7 %; max Delta_top 0 / 14.4 / 50.3 ueV (mu_C 0, E_Z 1.3).  The gapless share of Q = -1
  is 100 / 52 / 56 %; values are < 1e-3 or > 0.08 ueV, nothing between.  Pure Rashba (scharf_absTopo_muC_a10_tc*):
  0 % gapless at every tc.  At mu_C 0.7: gapless / 9.8 ueV, xi_M 2.30-7.5 um, max 2 um/xi_M 0.87 / 37.5 ueV, 0.28-7.2,
  7.03.  The deck summary first printed xi_M 0.26 um for tc 0.5/0.75 = the [leads] TRAP (257 nm bulk mode at gapless
  points); now masked in the printout and in slides_nov.slide9 (GAPLESS_UEV = 0.01; orange = Q -1 but gapless).
  OPEN: the (phi, E_Z) Q maps (e.g. 45.6 % at a 5) count gapless points as topological -- gap not checked there.
  [2026-11-12: study_tuning [maps] cell 3 checks Delta_top on every Q = -1 point of the a = 5 map (cache
  scharf_absMap_gap_mixed_a5, figure plots/maps_mixed_a5_gap.png); PAUSED at 168 / 579 points.  At a = 10, tc 1 the [angle]
  map gives a gapless share of Q = -1 of 11.2 % at theta 0.15.]
* TEMPERATURE (a 20, L 2, phi pi): best neutral kappa at 0/10/20/50 mK, tau 0.7: 0 at every T (no neutral peak).
  Best region (a 10, tau 0.5, L 2): along phi at E_Z 0.93, kappa = 1/2 +- 0.03 with |G| < 0.1 kappa over 1.37 / 1.07 /
  0 / 0 pi (max kappa 0.542 / 0.499 / 0.446 / 0.212); along E_Z at phi 1.742 pi: 55 / 30 / 0 / 0 ueV.

## SCHARF: the SOC ANGLE theta -- why the mixed SOC is gapless inside Q = -1, protection and plateau vs theta (2026-11-12/13, `study_mixed.py` [angle])
alpha = 16 cos(theta pi), beta = 16 sin(theta pi) meV nm (theta 0 = Rashba 16; thesis 0.15); W 100, L 2 um, mu_C 0.7, a 10;
transport tau 0.5.  Figures plots/angle_mechanism.png, angle_bulk.png, angle_maps.png, angle_transport.png,
angle_summary.png; caches scharf_absAngle_*; logs/study_mixed_angle.log (cells 1-3, complete), study_mixed_angle2.log
(cells 1b, 3b partial), study_mixed_angle3.log (2026-11-18: 3b and cell 4 complete; plots/angle_rotation.png).
* MECHANISM (cell 1, phi pi, E_Z 0.6): the middle BdG pair along k is e_+- = t +- g (t: tilt, g: half gap).  max |t| ~
  sin 4 theta (t / t(0.1 pi) 0.62-0.66 / 1 / 0.93-1.00 / 0.55-0.62 at theta 0.05 / 0.1 / 0.15 / 0.2; sin ratio 0.618 / 1 / 1 /
  0.618); g ~ cos 2 theta at tc 1 (g / g(0) 0.955 / 0.812 / 0.594 / 0.315 vs 0.951 / 0.809 / 0.588 / 0.309), faster at tc 0.5
  (0.926 / 0.731 / 0.480 / 0.233).  |t| > g = gapless at finite k with Q still -1: min (g - |t|) at tc 0.5 +11.9 / +5.6 /
  -0.09 / -3.1 / -2.7 ueV for theta 0 / 0.05 / 0.1 / 0.15 / 0.2; tc 1 stays gapped (+66.6 / +59.3 / +48.6 / +35.4 / +19.3).
  theta 0.25 (alpha = beta): t = 0 but g ~ 0 (node).  theta <-> pi/2 - theta is exact (0.35 = 0.15 to every digit).
* REFINED IN k (cell 1b: 401 k within +-4e-4 /nm of the minimum; the crossings can be ~1e-4 /nm wide): the coarse k grid
  got the sign wrong in 4 of 51 cases -- theta 0.15, tc 0.75 is GAPLESS at E_Z 1.2 (-0.47 ueV) and 1.4 (-5.1), not
  +4.4 / +0.9; alpha = beta gapless at tc 0.5 and 1.  theta 0.15, tc 0.75 vs E_Z: +3.5 (0.2), +9.8 (0.6), +2.0 (1.1),
  -0.5 (1.2), -2.9 (1.3), -5.1 (1.4) ueV.
* FIELD ROTATION (cell 1b, single points: theta 0.15, E_Z 0.6, phi pi; field d [pi] off theta_z,opt): tc 1: Delta_top
  +35.2 (d 0) -> +38.5 (-0.005), +36.0 (-0.010), +32.5 (-0.015), +22.1 (-0.03), +1.4 (-0.06), gapless at -0.1; +31.9 at
  +0.005.  tc 0.5: gapless at d 0 (-3.65) but GAPPED at -0.010 (+4.16) and -0.005 (+1.27), gapless from -0.015.
* FIELD ROTATION, full (cell 4, 2026-11-18).  phi = pi lines (gapped E_Z range; max Delta_top ueV), d = 0 / -0.005 /
  -0.010 / -0.015: tc 0.5 none (gapless) / 0.45-1.25, 1.7 / 0.05-1.30, 4.2 / 0.05-0.40, 0.7; tc 0.75 0.10-1.15, 9.8 /
  0.10-1.70 (all of Q = -1), 14.3 / 0.10-1.70, 15.5 / 0.10-1.35, 10.2; tc 1 always gapped, 37.5 / 38.5 / 38.0 / 33.9.
  (phi, E_Z) map, tc 1 (21 x 16), d = -0.010 / 0 / +0.010: Q = -1 45.4 % at all; gapped 44.1 / 40.3 / 36.8 %; gapless share
  of Q = -1 2.8 / 11.2 / 18.9 %; max Delta_top 81.6 / 80.9 / 74.6, mean 30.1 / 30.1 / 29.0 ueV.  Transport (tau 0.5, coarse)
  at d = -0.010 vs 0: kappa > 0.25 27.1 vs 27.8 % (neutral 0.83 both), plateau in gapped Q = -1 2.4 vs 2.5 %, longest run
  0.80 pi (E_Z 0.95) both.  => a ~1.8 deg rotation the right way removes 3/4 of the gapless strip (protection by AREA, not
  by gap size; at tc 0.75 also +50 % gap) and changes nothing for kappa = 1/2.  The other sense makes it worse.
* BULK vs theta (cell 2, phi pi, tc 1): max Delta_top 66.5 / 60.1 / 50.5 / 37.5 / 20.0 ueV for theta 0 ... 0.2; tc 0.5
  gapless from theta 0.10 on; tc 0.75 gapped to E_Z 1.30 / 1.15 / 1.10 (theta 0.10 / 0.15 / 0.20); theta 0.25 gapless at
  every tc.  (phi, E_Z) map (tc 1, E_Z 0-2 step 0.1 x 15 phases): Q = -1 47.9 / 46.7 / 46.7 / 45.4 / 43.5 %, gapped 47.6 /
  43.2 / 41.3 / 40.3 / 40.6 %, gapless share of Q = -1 0.7 / 7.5 / 11.6 / 11.2 / 6.6 %; max Delta_top 101 / 94 / 87 / 81 / 74
  ueV, always at (1.33 pi, 0.6) -- the maps are asymmetric in phi even for Rashba.  xi_M (phi pi, gapped): max L/xi_M 7.83 /
  7.79 / 7.63 / 7.03 / 4.00; L/xi_M >= 3 at E_Z 0.10-0.70 (theta <= 0.15), 0.20-0.50 (0.2).  theta 0.15 = the [oscillation]
  cache exactly.
* TRANSPORT vs theta (cell 3, coarse E_Z 0.05 x 30 phases): kappa > 0.25 on 16.2 / 18.7 / 22.7 / 27.8 / 29.5 % (neutral
  share 0.61 / 0.65 / 0.80 / 0.83 / 0.86); plateau in gapped Q = -1 4.1 / 3.7 / 2.8 / 2.5 / 2.3 %.  Fine lines (E_Z +-0.1 step
  0.01, 240 phases; Q, Delta_top every 4th phase): best run in gapped Q = -1 1.125 (E_Z 0.910) / 1.100 (0.910) / 0.667
  (1.050) / 1.367 (0.930) / 0.908 (1.060) pi; on the runs kappa 0.471-0.500, |G| <= 0.048, 2M residual <= 7.3e-4.  The best
  lines look alike at every theta (kappa ~ 1/2 over phi ~ 1-2 pi, dips near 0.25 and 0.75 pi).  Cell 3b (E_Z refined to
  0.0025, all in gapped Q = -1): 1.225 (0.9150) / 1.250 (0.9175) / 0.783 (1.0550) / 1.367 (0.9300) / 1.050 (1.0675) pi for
  theta 0 ... 0.2; for 0.05 and 0.20 the best sits on the edge of the +-0.0075 window (optimum may lie beyond).  The 0.15
  winner is itself field-sensitive (1.23 / 1.37 / 1.15 pi at 0.9225 / 0.930 / 0.935).
* READING: Rashba-like (theta -> 0) has the most protection (Delta_top 66.5 vs 37.5 ueV at 0.15, no gapless strip) but the
  plateau does not improve with it: the plateau % falls only slightly with theta and the best single line (1.05-1.37 pi
  for all theta but 0.10) is set by fine-tuning the field.  The field rotation likewise buys gapped area, not plateau.
  theta and the field angle are levers for protection, not for kappa = 1/2.
* OPEN: the 3b optimum beyond the window for theta 0.05 / 0.20; whether the mixed-SOC xi_poly / xi_M != 1 (dip redo)
  comes from the tilt.

## SCHARF: DOS(E, phi) vs probe bond, and EXCEPTIONAL POINTS of the open junction (2026-11-13/18, `study_tuning.py` [spectrum], [ep], [ep2]-[ep4])
Thesis junction a 10 (mixed SOC, mu_C 0.7, tc 1, W 100, L 2 um).  Formulas (poles, EP condition, overlap, 2x2
reduction): DERIVATIONS.md section 7.7.  Figures plots/spectrum_mixed_a10.png,
ep_mixed_tauL<t>[_tauR0.5].png, ep_fine_mixed_tauL<t>.png; caches scharf_absSpec_*, scharf_absEP_*.
* [spectrum] DOS(E = 0) survey at tau 0.5 (E_Z 0.30-1.30 step 0.025, eta 1e-4 meV, probes ATTACHED): brightest spot inside
  (eroded) Q = -1 at E_Z 0.325, phi 0.85 pi (370 /meV, median 41).  There Delta_top = 0 (Q = -1 but gapless at phi
  0.833-0.867 pi, k 0.0275 /nm: the band tilt of study_mixed [angle]) and transport is CHARGED (kappa 0.82 / |G| 0.75 at
  tau 0.5, 0.89 / 0.78 at tau 1).  DOS(E = 0) at the spot vs median over the Q = -1 phases: tau 0: 234 vs 527 (a DIP),
  tau 0.5: 245 vs 65, tau 1: 61 vs 14 (peaks) -> the bright spots exist only with the probes attached (reading, untested:
  the probes broaden the end Majoranas strongly, the extended gap-closing states much less).  DOS(E, phi) (E 0-40 ueV
  step 1, eta 0.5 ueV, tau 0 / 0.5 / 1): tau 0 = a sharp E = 0 line over phi 0.58-1.42 pi (the Q = -1 range) ending at the
  Q boundaries; the probes only broaden it.  DOS(-E) = DOS(E) to 5e-13.  ~11 s per energy (121 phases); 6 workers x 2
  BLAS threads were only ~1.3x faster than one process (the 121 per-phase 800 x 800 solves dominate).
* tools.nh_spectrum = poles of the G of transport / dos_total_phis: eigenvalues of H_C + Sigma_ribbons(0) + Sigma_L(tau_L)
  + Sigma_R(tau_R) with metric M = 1 - Sigma_rib'(0) (ribbons linearised around E = 0 -> generalised problem; probes at
  E = 0; tau_L, tau_R separate).  Checks at E_Z 0.93, pi: top-ribbon gauge rotation = direct ribbon to 7e-16; lossless
  pair +-0.407 ueV = closed box (study_oscillation.box_point) 0.40717; next level 57.60 vs box 57.20.  Frozen Sigma(0)
  (M = 1) puts EVERY energy x1.64 too high (0.668, 99.8 ueV; quasiparticle weight in the strip ~0.61): EP positions
  would survive (eps and Gamma scale alike), absolute energies not.  The pair = the two poles with the smallest |Re E|
  (ordering by |E| swapped in a bulk pole at tau_L 0.8: Im -45 vs Re 40 ueV).  ~8 s per phase, one thread.
* MODEL.  Li & Trauzettel, PRB 112, 184504 (arXiv 2507.23022): p-wave JJ with a loss -iV delta(x), Z << 1 ONLY (figures
  Z = 0.2); EPs at cos(phi/2) = +-Z around the MZM crossing at pi, poles purely imaginary between them, one mode sharp.
  Here the loss is the probe leads.  Two Majoranas with loss matrix [[G_L, c], [c, G_R]] (c: each probe also touches the
  OTHER Majorana's tail): E = -i Gbar +- sqrt(eps^2 - delta^2 - c^2), delta = (G_L - G_R)/2 -> EPs at |eps| =
  sqrt(delta^2 + c^2) =: delta_eff; between EPs Re E = 0; eigenvector overlap |<psi_1|psi_2>| = |eps| / delta_eff inside,
  delta_eff / |eps| outside: 1 at the EPs, 0 at eps = 0 (pure gamma_L, gamma_R).  [derived here, 2x2 algebra]
* RESULTS, E_Z 0.93 (phi = pi is a lossless box zero), 121 phases, then fine scans (0.001 pi; control 0.0002 pi) around
  every EP.  Lossless eps(phi): crossings ~0.47 and ~0.99 pi, max 9.2 ueV at 0.73 pi.  tau_R 0.1, EPs (phi / pi):
      tau_L   around the 0.47 crossing     around the 0.98 crossing     other
      0.1     0.4691-0.4699 (0.0008)       0.9877-0.9883 (0.0006)        (EQUAL bonds)
      0.2     0.4685-0.4765                0.9815-0.9885
      0.3     0.4645-0.4875                0.9705-0.9905
      0.4     0.4575-0.5055                0.9545-0.9945
      0.5     0.4395-0.5295                0.9315-1.0055 (phi = pi inside: poles -1.19i, -2.73i ueV)
      0.6     0.3805-0.5625                0.9015-1.0405                 1.6145-1.9595 (no crossing)
      0.7     merged: Re E = 0 except 0.6025-0.8645 (around the eps maximum)
      0.8     merged: Re E = 0 except 0.6475-0.8245
  Overlap 0.97-1.000 at every EP on the fine grid.  Each crossing opens into a segment bounded by two EPs (the paper's
  structure); the segment roughly doubles per 0.1 in tau_L.  H is 2 pi-periodic in phi, so the parity crossings come in
  pairs -> EP pairs in pairs (the paper: one crossing per 2 pi, 4 pi-periodic p-wave ABS, the outer Majoranas left out).
  - EQUAL BONDS ALSO HAVE EPs, ~0.001 pi wide (Im split -0.088 / -0.116 ueV; overlap 0.996 at an EP).  MIRROR CHECK:
    (tau_L, tau_R) = (0.1, 0.5) and (0.5, 0.1) give the same pair to 0.000 ueV at the same phi (6.3 ueV at 2 pi - phi):
    the junction is exactly left/right equivalent, delta = 0 for equal bonds, so those EPs come from c ~ 0.017 ueV
    (c / Gbar ~ 0.17, same order as e^{-L/xi_M} ~ 0.1-0.2 here -- order of magnitude only, not derived).  [An earlier
    reading 'Gamma_L / Gamma_R ~ 1.4, junction not symmetric' was WRONG.]  The 'delta' read off the overlap is delta_eff.
  - STRONG ASYMMETRY (tau_L >= 0.6 at tau_R 0.1): delta_eff exceeds eps also where eps has no zero -> loss-driven segment
    1.6145-1.9595 pi (overlap 1.000 at both EPs); at 0.7 / 0.8 the segments merge and a real splitting survives only around
    the eps maximum (damped-oscillator picture: overdamped wherever eps < delta_eff).  Not in Li & Trauzettel (Z << 1
    only; their closed form has 1/(1 - Z^2)), but KNOWN in open Majorana systems (literature check 2026-11-18, abstracts
    only): EP bifurcation from asymmetric coupling of the Majorana components to the reservoir (Avila, Penaranda, Prada,
    San-Jose, Aguado, Commun. Phys. 2, 133 (2019)); zero-energy pinning above a threshold coupling, one mode non-decaying
    (San-Jose, Cayao, Prada, Aguado, Sci. Rep. 2016); the transition becomes 'a flattened zero energy line bounded by
    exceptional points', very strong non-Hermiticity harmful (Cayao, PRB 110, 085414 (2024)); EPs vs phase and zero
    real-energy lines in JJs with normal leads, inner vs outer Majoranas (Cayao & Sato, arXiv 2307.15472, 2408.17260,
    JPSJ 95, 014705 (2026)); infinite-system vs S-matrix-pole transitions differ in finite wires (Pikulin & Nazarov,
    PRB 87, 235421 (2013); cf. det r = 0 vs EPs here).  Possibly specific here (full texts not read): the 2D planar
    junction with exact NEGF probes; equal-bond EPs from the cross term c alone in an exactly mirror-symmetric junction;
    det r (G_L G_R) vs EPs (difference) quantitatively; asymmetry does not improve the plateau.  The broad pole reaches 24 / 45 ueV (tau_L 0.7 / 0.8)
    > Delta_top 19 ueV at pi: the two-Majorana reading is at its limit there (the poles themselves are exact).
  - tau_R 0.5 sweep (coarse grid), range with Re E = 0: tau_L 0.1 / 0.2 / 0.3 / 0.4: 0.165 / 0.149 / 0.132 / 0.066 pi;
    0.6 / 0.7 / 0.8: 0.116 / 1.058 / 1.851 pi.  Shrinks towards tau_L = tau_R and reappears above, faster above (Gamma(tau)
    is steep: the width DIFFERENCE sets it, not |tau_L - tau_R|).  Segment centres move with the probes' level shift
    (0.51 / 0.94 pi at tau_R 0.5 vs 0.47 / 0.98 at 0.1).
  - det r and kappa at E = 0 depend only on G_L G_R in the Nilsson model (c = 0): (1, 1) and (4, 0.25) ueV give equal kappa
    and det r but only the second has EPs.  det r < 0 <=> eps < sqrt(G_L G_R) is a DIFFERENT boundary from eps = delta_eff
    (diagonal model: EP segments inside det r < 0 iff G_L / G_R < 3 + 2 sqrt 2 ~ 5.8).  The det r regimes of the symmetric
    data (lobes with eps_max above / below Gamma) are NOT EP regimes (EPs there ~0.001 pi).  EPs show in energy-resolved
    quantities (poles, DOS(E) line shape), not in E = 0 transport.
  - TWO-MAJORANA REDUCTION FROM THE FULL MODEL ([ep2], tools.nh_pair_2x2, plots/ep2_mixed.png, no fit): lossless pair ->
    left / right-localised combinations -> probe self-energies projected (first order).  Equal bonds 0.1 / 0.1: G_L = G_R
    EXACTLY along all phi (0.06-0.22 ueV), |c| 0.0005-0.014 ueV; inside the tiny segment (0.4695 pi) 2x2 poles -0.086 i /
    -0.114 i vs exact -0.087 i / -0.116 i -> the equal-bond EPs ARE the cross term (CONFIRMED).  Asymmetric: delta >> |c|
    (|c| <= 0.07 ueV at (0.3, 0.1) vs delta up to ~1).  EP positions, 2x2 vs exact: (0.3, 0.1) 0.4664-0.4847 / 0.9779-0.9865
    vs 0.4645-0.4875 / 0.9705-0.9905 (segments 20-60 % too narrow); worse with tau_L: at (0.5, 0.1) phi = pi falls outside
    the 2x2 segment but inside the exact one (2x2 total loss 2.62 vs exact 3.91 ueV there); the loss-driven segment (0.6)
    and the merging (0.7, 0.8) are ABSENT from the 2x2 (it keeps four EPs at the crossings).  -> first order in the probes
    holds only for tau_L <~ 0.3; beyond, the probe reshapes gamma_L and only the full poles are quantitative -- the
    strong-loss regime is non-perturbative.  (Equal bonds: the 2x2 has no sign change of D on the 121-phase grid only
    because those segments are ~0.001 pi wide.)  The mirror pairs give identical 2x2 eigenvalues.
  - det r / kappa WITH ASYMMETRIC PROBES (2026-11-16, [ep3], tools.transport_phis tau_R=; checked: tau_R = tau_L
    reproduces exactly; kappa, G identical under tau_L <-> tau_R, det r_L differs slightly).  plots/ep3_detr_mixed.png:
    det r = 0 and the EPs are DIFFERENT points.  (0.5, 0.1): det r_L never crosses 0 (fine 0.0005 pi: min +0.21 / +0.39 at
    the two crossings) although phi = pi etc. lie between EPs (det r there +0.29..+0.81); (0.6, 0.1): min +0.50.  So EP
    segments sit in det r > 0 at strong asymmetry.  The two-Majorana relation kappa = (1 - det r^2)/2 still holds to 0.011.
    The parity-crossing kappa dip is filled in: kappa one broad maximum 0.468 / 0.415 instead of 1/2 - 0 - 1/2 (reading,
    untested: the strong probe also reaches the other Majorana's tail = a direct path at eps = 0) -- but below 1/2,
    |G| / kappa 0.13-0.17 and kappa ~0.05 elsewhere (the small G_L G_R).  Weak asymmetry (0.4, 0.6), 801 phases
    (plots/ep3_fine_mixed_tauL0.4_tauR0.6.png): EPs 0.4563-0.5638 and 0.8962-0.9813 pi, overlap 0.996 at the EPs and
    ~0.01 at eps = 0 (the full W); both segments INSIDE det r < 0 (det r -0.67..-0.13 there; G_L / G_R < 5.8), but det r < 0
    also extends beyond them and appears where there is no EP (phi < 0.095, > 1.486 pi; det r = 0 at 0.095, 0.376, 0.574,
    0.890, 1.030, 1.486 pi); kappa 0.49-0.50 with |G| ~ 0 over most of phi, dips only in the EP segments (0.28 / 0.38) and
    at the eps maximum (0.11).
  - ASYMMETRIC BONDS AND THE PLATEAU (2026-11-16, [ep4], plots/ep4_plateau_mixed.png, ep4_EZcut_mixed.png): longest
    [plateau] run along phi (241 phases, Q = -1; scoring reproduces 1.367 pi for 0.5 / 0.5 at 0.930) over (tau_L, tau_R) in
    {0.3..0.7}^2 at E_Z 0.925 / 0.930 / 0.935: the long runs lie on the anti-diagonal tau_L tau_R ~ 0.21-0.25 (constant
    G_L G_R), symmetric under the swap; best 1.367 pi = (0.5, 0.5) at 0.930 = (0.3, 0.7) at 0.925; off the ridge ~0.
    Along E_Z (0.80-1.10, step 5 ueV) at phi 1.742 pi: windows 50 (0.5/0.5), 55 (0.6/0.4), 65 (0.7/0.3), 25 (0.6/0.5),
    25 (0.5/0.3) ueV; the ridge curves nearly coincide, (0.7, 0.3) wins only because its slightly smaller product puts the
    lobe top at det r -0.06 (x_max ~ 1) instead of -0.14 (the existing Gamma ~ (0.72-1) eps_max rule); a symmetric
    tau ~ 0.47-0.48 should do the same (not computed).  At phi = pi all windows 5-15 ueV.  => ASYMMETRY DOES NOT IMPROVE
    THE PLATEAU: it is a function of G_L G_R (Nilsson), asymmetry only re-tunes the product; the EP segments sit where the
    plateau already breaks (the crossing dips).
  - SECOND FIELD E_Z 0.70 ([ep3] cell 3, plots/ep_mixed_tauL<t>_EZ0.7.png): lossless eps 1.85 ueV at pi, max 7.6; FOUR
    crossings (~0.16, 0.58, 0.92, 1.38 pi; eps < 0.6 ueV over ~1.3 -> 0.25 pi).  Equal bonds 0.1 / 0.1: an EP segment
    resolved on the coarse grid at the slow 0.16 pi crossing.  (0.2, 0.1): segments 0.042-0.242 and 1.325-1.475 pi (0.35 pi
    vs 0.017 pi at E_Z 0.93); (0.3, 0.1): merged 1.258 -> 0.325 pi through phi = 0 plus small ones at the steep crossings
    (1.12 pi in all), real splitting only around the three eps maxima.  Range with Re E = 0 for tau_L 0.1 ... 0.8:
    0.017 / 0.35 / 1.12 / 1.34 / 1.65 / 1.79 / 1.82 / 1.87 pi; from tau_L 0.6 only the window around the big eps maximum
    keeps a real splitting (EPs 0.658-0.875 / 0.675-0.858 / 0.692-0.825 pi for 0.6 / 0.7 / 0.8) and phi = pi is between EPs
    (poles -0.95 / -5.96 i, -0.62 / -11.35 i, -0.44 / -20.7 i ueV: the narrow one sharpens as the broad one takes the
    loss).  The segment width is set by |d eps / d phi| at the crossing; merging needs delta_eff > eps over a range, so it
    sets in at weak probes where eps is small (lower field: tau_L 0.3 here vs 0.7 at E_Z 0.93).  PAIR-SELECTION TRAP:
    at E_Z 0.70 the bulk levels sit at |Re E| 25-40 ueV, so at tau_L 0.8 the broad Majorana pole (Im ~ -48) was not among
    the k = 8 poles closest to 0 and a bulk pole was taken as 'pole 2' (fake EPs at 0.142 / 1.775 pi); fixed by k = 16,
    keeping the 8 with the smallest |Re E| (ep_point k=).  Check any new strong-loss run for this.
  - FULL-TEXT READ (2026-11-18, via the arXiv / ar5iv HTML, tool summaries): Cayao & Sato 2307.15472 -- minimal JJ models,
    leads as -i Gamma (Gamma_L != Gamma_R or a spin-polarised lead on N), Rashba S-N-S nanowire; EPs vs phi, Re E pinned
    between EPs, supercurrent enhanced at zero-energy EPs; EPs both below and above B_c, and below B_c 'these EPs do not
    depend on the length of the S regions'; no DOS / conductance, no topological-vs-trivial claim.  Cayao & Sato
    2508.19433 -- four-MZM effective model, leads on outer or inner MZMs; zero-real-energy lines bounded by EPs near
    phi = pi, and with one outer lead off EPs also near phi = 0 (a longer zero line -- cf. our merging); local / nonlocal
    spectral weight largest between EPs; length only via the overlap t'/t.  Pikulin & Nazarov PRB 87, 235421 -- 'pole
    transition' = change of the number of purely imaginary S-matrix poles, offset from the bulk transition by ~ v/L, can
    occur in the trivial region; one lead; conductance peaks approach zero bias but 'never merge' (narrow ZBP on a wider
    dip).  => OPEN and possibly new: EP-segment width vs L (end Majoranas: eps ~ e^{-L/xi_M} -> exponential growth; the
    trivial-regime EPs above are L-independent); finite-bias local / nonlocal conductance with two asymmetric probes in a
    realistic geometry; where EPs occur relative to Q and the gapless strip.
  - EP SEGMENT vs LENGTH (2026-11-18, [ep5], (tau_L, tau_R) = (0.3, 0.1)).  ALONG phi at E_Z 0.93 (plots/ep5_length_mixed.png):
    NOT a clean test -- the crossings move with L (eps ~ cos(k_osc L + phi0)): L 1.5 at 0.133 / 1.333 pi, L 2 at 0.467 / 0.983,
    L 2.5 NONE (|eps| >= 1.66 ueV); total phase between EPs 0.099 / 0.050 / 0 pi; delta_eff 0.81 / 0.42 ueV at the L 1.5 / 2
    crossings (different phases -> different G_L, G_R, c); all crossings gapped (Q = -1, Delta_top 17-51 ueV, xi_M 0.74-1.49
    um).  ALONG E_Z at phi 1.742 pi, the zero nearest 0.90 meV (plots/ep5_lengthEZ_mixed.png):
        L [um]   zero [meV]   width [ueV of E_Z]   2 delta_eff/|d eps/d E_Z|   delta_eff [ueV]   |d eps/d E_Z| [ueV/meV]   xi_M [nm]
        1.5      0.905         12.4                 13.1                         1.04               159                      629
        2.0      0.850         42.0                 39.4                         1.06                54                      547
        2.5      0.945         21.0                 24.4                         0.99                81                      695
        3.0      0.895        >= 243 (window)      133                           1.08                16                      613
    The local law width = 2 delta_eff / |d eps / d E_Z| holds to 5-15 % (L <= 2.5); delta_eff is L-independent here (set by
    the probes); L enters only through the slope, which falls ~10x from 1.5 to 3 um (e^{1.5 / 0.62} ~ 11); fit xi 607 nm vs
    xi_M 621 nm (no fit), but lobe-to-lobe scatter ~2x (the matched zeros sit in different lobes; L 2.5 below the trend) and
    L 3 is a lower bound (eps < delta_eff over a whole lobe: segments merge).  => the pinned range grows ~ e^{L / xi_M}, a
    property of END Majoranas (cf. the L-independent trivial-regime EPs of Cayao & Sato) -- consistent, not a precision test.
  - ENTROPY BETWEEN EPs (2026-11-18, [ep6], plots/ep6_entropy_mixed.png; SIDE STUDY, user: not thesis relevant).  Pino,
    Meir & Aguado (arXiv 2405.02387) predict S = (1/2) ln 2 between EPs (one pole sharper than kT, one broader), visible
    via dS/dphi = -(hbar/2e) dI/dT.  S from the full-model pair poles (tools.pole_entropy, Lloyd DOS, checked limits).
    Single-probe lines (tau_R = 0, plots/ep_mixed_tauL<t>_tauR0.png): at phi = pi poles -0.97i / -2.84i (0.5, 0) vs
    -1.19i / -2.73i (0.5, 0.1); -1.38i / -19.24i (0.8, 0) vs -1.49i / -19.23i (0.8, 0.1) ueV -> the sharp pole's width
    comes mainly from the LEFT probe itself (grows with tau_L), not from the right one.  Result: NO (1/2) ln 2 plateau;
    S rises smoothly through it (0.8, 0) at pi: 0.06 / 0.24 / 0.53 / 0.81 at 1 / 7 / 37 / 206 mK (a -0.01i / -20i
    test pair holds 0.48-0.56 over 1-20 mK); S and dI/dT smooth through every EP; the pair's dI/dT drops from +-0.25
    nA/mK (closed, 3 mK) to +-0.015 (tau_L 0.8).  The planned exact supercurrent (Li & Trauzettel's CPR) was dropped.
  - THERMAL TRANSPORT ACROSS THE EP SEGMENTS (2026-11-18, [ep7], plots/ep7_kappaE_mixed_tauL<l>_tauR<r>.png; full model:
    kappa(E, phi) from tools.channel_point (now with tau_R), 0 + 60 log energies 0.01-70 ueV, kappa(T) by tools.thermal).
    E* = energy of the maximum of kappa(E):
        pair        E* = 0 in / out of segments   kappa max 0 / 10 / 20 mK   kappa(20)/kappa(0) median in / out
        (0.3, 0.1)  100 % / 1 %                   0.488 / 0.077 / 0.064      0.004 / 0.134
        (0.5, 0.1)  100 % / 5 %                   0.448 / 0.075 / 0.059      0.021 / 0.841
        (0.8, 0.1)  100 % / 36 % (seg. 1.82 pi)   0.280 / 0.077 / 0.033      0.134 / --
        (0.4, 0.6)  91 % / 85 %                   0.531 / 0.469 / 0.390      0.254 / 0.711
        (0.5, 0.5)  33 % / 49 % (seg. 0.05 pi)    0.542 / 0.497 / 0.449      0.350 / 0.823
    Outside the segments, where the splitting is resolved, E* follows |Re lambda| of the full-model pair poles (just below
    it).  Inside the segments kappa(E) is a zero-energy peak as narrow as the sharp pole -> the EP SEGMENTS ARE THE
    T-FRAGILE PART OF kappa(phi) for every pair.  Weak or strongly asymmetric probes: the large kappa (0.45-0.49) exists
    ONLY inside the segments at T = 0 and is gone by 5-10 mK.  Strong balanced probes ((0.4, 0.6), (0.5, 0.5)): widths
    exceed the splitting over most of phi (kappa(E) peaked at 0 also outside the segments), the kappa ~ 1/2 plateau lies
    OUTSIDE the segments and is the T-robust one (0.39-0.45 at 20 mK); the segments sit in the crossing dips.  Equal bonds
    0.5 / 0.5 have RESOLVED segments (0.05 pi total; ~0.001 pi at 0.1 / 0.1): they grow with the overall strength too.
  - EP SEGMENTS AND THE kappa PLATEAU ALONG E_Z vs THE PROBE COUPLINGS (2026-11-18, [ep8], plots/ep8_coupling_mixed.png,
    ep8_lines_mixed.png, logs/study_tuning_ep8_analysis.log; full model, phi 1.7417 pi, E_Z 0.80-1.10 meV step 5 ueV,
    36 bond pairs from {0, 0.1..0.6, 0.8}, E = 0, T = 0).  Lossless crossings of eps(E_Z) at 841.7 / 1026.7 ueV
    (|d eps/dE_Z| = 0.058 / 0.198).  One-probe pair width Gamma_1(tau) at the crossings: 0.23, 0.97, 2.36, 4.72, 8.64,
    15.2, 47.6 ueV for tau 0.1..0.6, 0.8 (~tau^2 up to 0.4, steeper beyond); two-probe widths add (a split pole has
    -Im lambda ~ SigmaGamma / 2, within the ~10 % drift of Gamma_1 between the two crossings).
    LENGTH LAW: an EP segment along E_Z spans L_EP = (1.07 +- 0.10) |DeltaGamma| / |d eps/dE_Z|, DeltaGamma =
    Gamma_1(tau_L) - Gamma_1(tau_R) (17 untruncated segments, both crossings; up to 1.37 for the longest, where eps(E_Z)
    curves) -> set by the width DIFFERENCE, not by the overall strength.  SigmaGamma only shifts the segment to lower
    E_Z (by 1-23 ueV, growing with the stronger bond) and leaves a short residual segment at equal bonds (0.4/0.4 1.7,
    0.5/0.5 9, 0.6/0.6 19, 0.8/0.8 51 ueV total).  tau_L = 0.8 with tau_R <= 0.5: the whole 300 ueV window is one segment.
    kappa INGREDIENTS: G ~ 0 along E_Z (|G| <= 0.025 for the four pairs of the lines figure), T_ee ~ T_he ~ kappa / 2; the lossless crossings are kappa dips whose
    depth is set by the asymmetry (tau_L 0.5, crossing 1027: kappa 0.13 / 0.15 / 0.20 / 0.29 / 0.44 for tau_R 0.5..0.1;
    at equal bonds the dip deepens with strength: 0.19 / 0.16 / 0.10 / 0.05 for 0.2 / 0.4 / 0.6 / 0.8).
    kappa = 1/2 PLATEAU ALONG E_Z (longest |kappa - 1/2| < 0.02 window; tol 0.01 / 0.03 in brackets) -- two routes:
      (A) balanced probes, SigmaGamma ~ 10-30 ueV: a dome between the crossings OUTSIDE the segments, pair split with both
          poles equally broad: (0.5, 0.4) 85 [30 / 95] ueV, widths 6.2-6.5, |Re lambda| 4.9-6.9; (0.5, 0.5), (0.6, 0.4),
          (0.6, 0.6) 40 ueV, widths 8-14.  The (tiny) segments sit in the crossing dips.
      (B) strong + weak bond (0.8, 0.2): 85 [80 / 90] ueV entirely INSIDE one segment, carried by a narrow pole pinned
          at E = 0 (width 1.6-2.1 ueV) next to a 44 ueV one; (0.8, 0.4) 35 ueV also inside (narrow pole 7.5-12 ueV).  By
          [ep7] a 2 ueV carrier is the T-fragile kind (k_B x 20 mK = 1.7 ueV) -- finite T along E_Z NOT yet computed.
      Too weak (SigmaGamma < 5 ueV): <= 10 ueV.  Too strong (0.8 / 0.6, 0.8 / 0.8): kappa max 0.45 / 0.25, no plateau.
      The tol-0.03 windows are threshold-sensitive (the domes peak at 0.497-0.499): (0.6, 0.3) 100 -> 35 -> 20 ueV.
  - [ep9] (kappa(E, E_Z) for four pairs) STOPPED by the user 2026-11-19 at 12 / 61 energies (all < 0.05 ueV; resumable
    cache scharf_absEP9_ch_*).  Real cost ~42 min per energy on 6 one-thread workers (~7.5 h), not the 1.5 h estimated.
  - DOES THE ASYMMETRY (EP PINNING) PROTECT kappa? (2026-11-19, [nh], PLOTS ONLY from the [ep7] / [ep8] / study_mixed
    [deck] 3 caches; plots/nh_phi_mixed.png, nh_field_mixed.png, logs/study_tuning_nh.log).  NO:
      (1) T = 0 plateau along E_Z (phi 1.7417 pi, 26 two-probe pairs): the |kappa - 1/2| < 0.02 window follows
          sqrt(G_L G_R) (one-probe widths on the window) whether it lies between EPs or not; optimum ~6-6.5 ueV:
          (0.5, 0.4) 85 ueV outside segments = (0.8, 0.2) 85 ueV all between EPs; (0.8, 0.3) between 20 = (0.6, 0.5)
          outside 20; (0.8, 0.4) 35 vs (0.6, 0.6) 40.  Only (0.8, 0.2) is flatter at tol 0.01 (80 vs 30) -- one case.
      (2) Along phi (E_Z 0.93): the narrower pole between EPs has median width 0.5-2.8 ueV vs 3.3-7.4 outside (k_B T =
          0.86 / 1.72 ueV at 10 / 20 mK); kappa(20 mK) / kappa(0) median 0.02-0.35 between vs 0.71-0.84 outside;
          (0.8, 0.1) is 91 % between EPs and has kappa max 0.28 / 0.03 at 0 / 20 mK.
      (3) The probes leave the ribbons (Q, Delta_top) unchanged; Re lambda = 0 also where eps has no zero ((0.6, 0.1)
          1.61-1.96 pi) -> the pinning is set by the coupling asymmetry, not by topology.
    EQUAL bonds 0.5 / 0.5 along E_Z 0.85-1.05 meV at 1.7417 pi (deck cache = [ep8] at T = 0 to 1.2e-14): kappa max
    0.496 / 0.486 / 0.411 / 0.189 at 0 / 10 / 20 / 50 mK; |kappa - 1/2| < 0.02 window 38 / 3 / 0 / 0 ueV; kappa > 0.45
    run 63 / 50 / 0 / 0 ueV -> the E_Z plateau at this phase is NOT 20 mK robust even for balanced probes (along phi at
    E_Z 0.93 kappa max is 0.449 at 20 mK).  Not cached: kappa(E_Z, phi) and finite T along E_Z for unequal bonds; mu_C
    only for equal bonds (plots/tuning_kappa_tau0.5.png).
  - WHY (2026-11-19, [nh] cells 3-4, plots/nh_why_poles_mixed.png, nh_detr_types_mixed.png; all full-model tests, no
    model assumed).  Particle-hole symmetry puts the pair poles at +-E0 - i gamma (outside the EPs) or at -i w+, -i w-
    (between them).  (1) E = 0 sees only their PRODUCT: det r_L = 1 - 2q, q = Gamma_L Gamma_R / |l1 l2| (Gamma = one-
    probe pair width), to <= 0.04 for all [ep8] pairs except 0.6-0.8 equal-strong (0.8/0.6 0.066; 0.8/0.8 FAILS,
    |l1 l2| < Gamma_L Gamma_R there); kappa = (1 - det r^2)/2 to 0.019.  The product is smooth through an EP -> kappa(E = 0)
    cannot see the EPs.  (2) kappa(E)/kappa(0) = |l1 l2|^2 / |(E - l1)(E - l2)|^2 (two nearest poles) to median 1e-4
    ([ep7], E <= 20 ueV; max 0.25 at a few balanced-pair phases).  (3) k_B T_half = 0.35 x min|lambda| (p10-p90
    0.28-0.49, 5 pairs x 121 phases).  Outside the EPs min|lambda| = sqrt|l1 l2|; between them it is w- = |l1 l2| / w+,
    pushed toward 0 as the strong probe's broad pole takes the loss.  At kappa = 1/2 (|l1 l2| ~ 2 Gamma_L Gamma_R):
    route A (0.5, 0.4) poles +-6.93 - 6.42i, route B (0.8, 0.2) -2.13i / -44.5i ueV at E_Z 945 -> same product, nearest
    pole 9.4 vs 2.1 ueV -> kappa halved near 38 vs 9 mK (RULE, not computed).
    TWO kappa = 1/2 TYPES (det r crossing 0 vs det r peaking below 0): ONE rule, det r = tanh(ln x), x = eps_eff /
    sqrt(Gamma_L Gamma_R), eps_eff^2 = |l1 l2| - Gamma_L Gamma_R (probe-shifted splitting; the lossless eps fails near the
    crossings).  |kappa - 1/2| < 0.02 <=> |det r| < 0.2 <=> x in 0.82-1.22.  det r peaks where eps_eff peaks (the lobe top
    between the crossings).  Windows along E_Z: lobe flank, det r crosses 0 once: 46, median 10 ueV (max 40); lobe top
    below 0: 5, max 40 ((0.5,0.5), (0.6,0.4), top -0.14 / -0.12); lobe top just above, crosses twice inside one window: 2,
    85 ((0.5,0.4) +0.17, (0.8,0.2) +0.12).  Longest window vs det r at the lobe top peaks at det r_top ~ 0 to +0.17,
    35 at +0.23, <= 25 beyond +0.45, 0 below -0.3.
    Inside vs outside: kappa = 1/2 at eps = sqrt(G_L G_R), EP at eps = |G_L - G_R|/2 -> the resonance lies inside a
    segment iff G_L / G_R > 3 + 2 sqrt 2 ~ 5.8; the [ep8] windows follow it except within ~15 % of the threshold
    ((0.8, 0.5) r 5.5 already inside -- the cross term c raises the EP threshold).  Clean one-message figures of all of
    this: study_tuning [clean], plots/clean/c1-c8.
  - LONG JUNCTIONS AT THE TRANSITION (2026-11-19, [trans], plots/trans_mixed.png; phi = pi, E = 0, L 2 / 4 um; L 8 um
    postponed by the user).  Bulk: E_c = 1.3613 meV (Q flips 1.3600 -> 1.3625; gap 1.24 -> 0.27 ueV, xi_M 10 -> 15.5 um
    over 1.335-1.360); the trivial side is GAPLESS.  Tunnel probes (tau 0.3387): kappa <= 0.02 within 0.1 meV of E_c and
    ~0.003 on the trivial side; the last kappa ~ 1/2 peaks sit 40-130 ueV below E_c and are charged (G/kappa 0.43).
    Transparent probes (tau 1): a sharp peak AT E_c, 0.84 (L 2) / 0.63 (L 4), charged (G/kappa 0.92 / 0.78); the gapless
    trivial side conducts (kappa 0.45-0.72, |G| 0.3-0.5); (1/2) sech^2(L/xi_M) with the bulk xi_M does NOT describe it.
    => no neutral universal kappa_0/2 at this transition: it opens into a gapless phase, so ordinary channels carry
    charge and heat there (Akhmerov's kappa_0/2 needs gapped phases on both sides).
  - BARRIER vs BOND PROBES (2026-11-19, [barr], plots/barr_mixed.png; equal probes, one-site lam_barr 260/220/180/150/120
    meV nm with tc_barr 1 = Gamma_1 3.1 / 4.2 / 6.2 / 8.6 / 12.7 ueV, phi 1.7417 pi along E_Z).  det r = 1 - 2q holds to
    <= 0.042 (bonds 0.066 w/o 0.8/0.8); Gamma_1(E_Z) drifts like the bonds'; plateau windows 15 / 20 / 75 / 15 / 30 ueV --
    on the bond curve (optimum sqrt(G G) ~ 6.2 -> 75 vs bonds 85), except lam 150 (15 vs bond 0.5/0.5's 40: its dome peaks
    at 0.47 < 0.48, the threshold sensitivity).  Level shift: kappa minima (crossings) 820-825 / 1005-1015 ueV vs
    825-840 / 1010-1025 for bonds of similar Gamma: a barrier shifts ~5-15 ueV more at equal Gamma.  => the contact model
    changes positions, not the relations or the plateau physics.
  - CHERN RIBBONS: A REAL PLATEAU (2026-11-19, [chern], plots/chern_maps_mixed.png; a 20, L 2 um, in-plane E_Z 0, out-of-
    plane Vz in the SC ribbons only, ribbon Chern transition 0.969 meV).  tau 1, phi = 0: neutral |kappa - 1/2| < 0.03 for
    EVERY Vz from 0.98 to 2.0 meV (10 grid points) at mu_C -0.4 ... 0.0, 1.0-1.75 at +0.1, 1.02-1.75 at +0.2; it starts at
    the Chern transition and fades by 2.5 meV (Chern gap 43 -> 37 ueV).  phi = pi: no plateau (kappa 0.5-0.98, scattered
    hits); tau 0.5: the phi = 0 plateau is GONE (kappa(0) ~ 0.05-0.4).  => a kappa = 1/2 plateau over a 1 meV range of a
    non-phase knob, tied to the ribbons' Chern phase -- but only with transparent probes, a depleted centre, phi near 0, and
    Vz ~ 1-2 meV out of plane in the SCs (3.5-7 T at g 10, or an exchange field).  NOT yet checked: a = 10, L-independence,
    temperature, the probe-transparency threshold, why phi = pi differs (mechanism not analysed).
    CORRECTION (same day, scratch kappa(phi) scan, 41 phases, tau 1, mu_C -0.2, a 20): NOT phi = 0 only -- kappa = 0.48-0.49
    with G = 0 at EVERY phase except a window narrower than +-0.05 pi around pi (kappa(pi) 0.89 at Vz 1.5, 0.68 at 1.05);
    Vz 0.9 (below the Chern transition): kappa = 0 at all phases.  The map only sampled phi = 0 and pi.  Reading (untested):
    with no in-plane field the strip's Andreev states reach E = 0 at phi = pi and add channels; elsewhere they are gapped and
    only the ribbons' chiral Majorana carries heat.
  - DARK STATES? (2026-11-19, [dark], plots/dark_mixed.png, logs/study_tuning_dark.log; San-Jose, Cayao, Prada &
    Aguado, Sci. Rep. 6, 21427 (2016), arXiv:1409.7306, read in full).  Their Gamma_0 -> 0 needs a single-mode helical
    reservoir, a transparent contact, the Andreev limit, and both Majoranas of a TRIVIAL pair at ONE contact; their poles
    are eig(H_S + Sigma_N(0)) = our probe treatment.  Here: phi 1.7417 pi, one probe (tau_R 0) or tau_R 0.1, standard probe
    (mu_N 3, no field: 4 open BdG channels) vs 'helical' probe (tools lead= hook: mu = mu_C 0.7 + the strip's in-plane
    field -> 2 open BdG channels = one spin-polarised mode; field-free mu 0.7 is closed).  tau_L >= 0.9: PAIR-SELECTION
    failure (broad pole outside k = 16) -- discarded.  Results (tau_L <= 0.8):
      lobe top (955 ueV, eps 7.49): pinned from tau 0.6 (std) / 0.7 (hel); narrow 6.32 -> 2.18 -> 1.16 (std), 4.85 ->
        3.18 -> 1.28 (hel) with |l1 l2| / eps^2 = 0.76-0.93 ~ 1 -> narrow ~ eps^2 / broad: the paper's law holds.
      crossings (840 / 1025, eps 0.10 / 0.34): narrow falls to a minimum at tau 0.3 (0.002 / 0.015 std, 0.002 / 0.009 hel),
        then RISES ~ the broad width (narrow / broad 6e-4 / 9e-3 std, 1.4e-3 / 1.7e-2 hel); |l1 l2| / eps^2 up to 137 (std) /
        171 (hel) -> a second loss channel on the pair, the far Majorana's tail at the probe (std 840: 6e-4 vs e^{-2L/xi_M}
        ~ 7e-4).  The helical probe does NOT remove it (at the exact zeros its floor / broad equals the standard one; the
        "~2x larger" at 840 / 1025 was the residual eps^2 / broad off the zero, larger for the smaller helical broad width).
      two probes: narrow -> Gamma_R of the second probe (0.23 std, 0.13-0.18 hel at 840) at every tau_L.
      one probe, crossings: local Andreev at E = 0 peaks (1.0-1.3 e^2/h at tau 0.3) and falls to ~0 at strong coupling --
        consistent with the paper's zero-bias DIP of width Gamma_0 on a 2e^2/h background (E = 0 inside the narrow pole).
      ENERGY-RESOLVED (cell 4, plots/dark_L_dip_mixed.png (b); E_Z 840, one probe, eta_rel 1e-7 = 2.5e-5 ueV, no failed
        solves): local Andreev = 2.00 e^2/h from ~0.1-1 ueV up to ~the broad width, dropping to A(0) = 0.000-0.009 at
        E = 0.  Half-max energy 0.0013 / 0.0074 / 0.031 ueV (std tau 0.3 / 0.5 / 0.8), 0.0007 / 0.013 / 0.055 (hel) vs the
        narrow pole width 0.002 / 0.008 / 0.030 (std), 0.002 / 0.013 / 0.050 (hel): dip width = narrow pole width
        (tau 0.3 a factor ~2 off: near the eps^2 regime, |l1 l2| / eps^2 ~ 0.2-0.5, and a 0.12-decade E grid).  = the
        paper's Fig. 4h.  A(0) ~ 0 means the probe sees BOTH Majoranas at E = 0; as e^{-2L/xi_M} -> 0 the dip closes and the
        single-Majorana 2e^2/h zero-bias value remains.
      FLOOR vs L (cell 3, plots/dark_L_dip_mixed.png (a); one probe, zero of each L tracked over 31 E_Z points, 1.5 ueV
        steps; parabola fits to the 5 nearest points give the same minimum within 2% -> NOT grid-limited):
        floor / broad vs e^{-2L/xi_M} (bulk xi_M at that zero, no fit) for L 1.5 / 2 / 2.5 / 3 um:
          ratio std tau 0.8: 0.53 / 0.50 / 0.69 / 1.10;  hel tau 0.8: 0.49 / 0.47 / 0.66 / 1.17;  std tau 0.5: 0.68 / 0.79 /
          0.97 / 2.99.  Two decades, INCLUDING the non-monotonic step at 2.5 um (its zero has the longer xi_M 695 vs 547
          nm), followed by all three series.  Open: tau 0.5 drifts up with L (x3 at 3 um) -- an extra contribution that
          falls slower than e^{-2L/xi_M} and matters more at weak coupling; not resolved.
    => the paper's mechanism is present (eps^2 / broad suppression at the lobe tops) but our junction is topological with
    the two Majoranas at OPPOSITE ends: at the splitting zeros the far Majorana's own tail at the probe gives it a direct
    width ~ broad x e^{-2L/xi_M} (confirmed vs L, no fit, tau 0.8), which GROWS with coupling -- that is why Im E does not
    climb to 0.  A single-mode (helical) lead does not remove it.  For two-probe kappa it is irrelevant as long as
    Gamma_R >> floor (0.13-0.23 vs ~0.003-0.02 ueV at L 2 um).
  - det r ~ 0 AND THE kappa PLATEAU ALONG phi (2026-11-23, [detphi], plots/detphi_mixed.png, logs/study_tuning_detphi.log;
    PLOTS ONLY: 321 deduplicated phase lines = study_mixed [plateau] caches scharf_absPlat_tr_mu0.7_* (equal bonds 0.4-0.7,
    L 1.5-3) + [ep4] lines (25 pairs x 3 fields); decomposition from the [ep] poles at E_Z 0.93).  Windows |kappa - 1/2| <
    0.02 with |G| < 0.1 kappa: 430, ALL obey kappa = (1 - det r^2)/2 to 0.001 -> along phi too, kappa ~ 1/2 <=> |det r| <
    0.2.  By what det r does in the window: crosses 0 once 313 (median 0.058 pi), turns just below 0 45 (0.26), turns just
    above 0 (two crossings) 30 (0.67), stays above 0 35 (0.08), wanders around 0 (3+ crossings) 7 (1.13 pi, max 1.175) --
    the E_Z rule (a single crossing is short; long needs det r stationary near 0) holds.  The 'det r at the lobe top'
    scalar does NOT predict the phi windows (lobes defined by det r >= -0.5 merge a flat near-0 stretch with a tall bump:
    tops > 0.45 still carry windows up to 1.1 pi; the longest has its top at +0.22, poking just out of the band).
    MECHANISM of the longest (E_Z 0.93, 0.5 / 0.5, 1.250 -> 2 pi -> 0.417 pi; also (0.4, 0.6), (0.3, 0.7) = the [ep4]
    anti-diagonal): the lossless splitting lobe between the zeros at ~0.99 and ~0.47 pi (through phi = 0) is FLAT, 6.5-7.2
    ueV over ~1.2 pi, and the one-probe width Gamma(phi) = 4.3-8.4 ueV matches it -> x = eps_eff / Gamma 0.78-1.21 across the
    window; near the 0.47 pi zero the probes also lift the splitting (lossless 1.7 -> eps_eff 3.3 ueV), stretching the window
    toward it.  The other lobe (0.5-0.95 pi): eps up to 9.2 vs Gamma 2.2-2.6 -> x 2.8, kappa 0.11.  Pole det r = 1 - 2
    Gamma^2/|l1 l2| vs transport: median 0.015, <= 0.058 in the window, but 0.3-0.4 AT the two zeros (0.53, 0.92 pi; tau 0.5
    / 0.5 -- worse than the <= 0.04 along E_Z; not analysed).  => the phi plateau is the same matching condition eps_eff ~
    sqrt(Gamma_L Gamma_R) as along E_Z, long only because this lobe is flat in phi; no protection involved.
  - EP SPECTRUM ALONG THE FIELD + TRANSPORT CHECKS (2026-11-23, [epfield], plots/epfield_mixed.png,
    epfield_kappaT_mixed.png, epfield_checks_mixed.png, logs/study_tuning_epfield3.log; user: 'EP spectrum vs Zeeman
    field first', before the Li & Trauzettel reproduction).  phi 1.7417 pi, E_Z 0.45-1.30 meV step 5 ueV (Q = -1 for
    ~0.53-1.2375), pair poles for (0, 0) and the Li-Trauzettel-like set-up 'one strong lossy probe + weak detector'
    (tau_L 0.3 / 0.5 / 0.7 / 1.0, tau_R 0.1) + E = 0 transport.  POLE SEARCH: the tau 1.0 broad pole (-163.5i ueV at 0.93)
    is beyond MANY bulk poles (|Re| 47-51) -> not among even the 32 closest to 0; tools.nh_spectrum / ep_point got an
    optional shift-invert target sigma (ueV; default 0 = unchanged); tau 0.7 / 1.0 searched around -40i / -150i, k 24.
    PAIR VALIDITY: next to the transitions a gap-closing bulk state (width 0.005-0.07, decoupled from the probes) has the
    smallest |Re E| and is taken as 'the pair' (tau 0.3 at 1220: 4.41 - 0.009i, Majorana pair at +-6.9 - 1.02i) ->
    masked: within 10 / 25 ueV of the lower / upper transition or two-pole kappa(E) > 1 (555, 560, 1215-1235 ueV).
    Lossless zeros 550.5, 643.1, 841.7, 1026.7, 1203.6 (1273.2 trivial side), |d eps/dE_Z| 0.017 / 0.006 / 0.056 / 0.193 /
    0.276: small and slow near the lower transition (long xi).  Fraction of the topological range between EPs: 33 / 58 /
    84 / 100 % for tau_L 0.3 / 0.5 / 0.7 / 1.0 -- segments merge from the slow low-field zeros upward; tau 1.0 has NO EP in
    the whole range (the Li-Trauzettel Z >= 1 situation), broad pole 100-170 ueV.  Segment law 2 delta_eff ~ L_EP x slope
    = 2.2 / 1.8 ueV (tau 0.3 at 842 / 1027) and 7.7 (tau 0.5 at 1027) ~ DeltaGamma ([ep8]).  Narrow width between EPs median
    0.23-0.46 ueV (~ Gamma_R of the detector).
    kappa(E), kappa(T) FROM THE POLES (cell 3): with the weak detector kappa is gone by 5 mK: max 0.19 / 0.16 / 0.15 / 0.14
    at 5 mK, 0.07-0.04 at 20 mK; kappa(20)/kappa(0) median 0.006-0.04 between EPs vs 0.08-0.74 outside.  (kappa = 1/2 needs
    eps_eff ~ sqrt(G_L G_R) with G_R ~ 0.2 ueV -> a ~0.2-1 ueV pole -> k_B T scale of a few mK: set by the coupling, not a
    sign of 'no MZM'.)
    FULL-TRANSPORT CHECKS (cell 4; 10 field / bond points, 49 energies 0-100 ueV, eta_rel 1e-7, 0 unreliable solves;
    ~1.5 h on 6 workers, not the 25 min estimated): the two-pole kappa(E) matches the full model to <= 0.009 of kappa(0)
    (0.024 at a kappa(0) = 0.009 point) for E <= 20 ueV outside, AT and between EPs, tau 0.3-1.0; kappa(T) full = poles to
    3 decimals (0-50 mK).  AT THE EP (tau 0.3 at 816.04, poles +-0.03 - 1.31i; tau 0.5 at 999.32) kappa(E) is the squared
    Lorentzian w^4/(E^2 + w^2)^2 to 0.000 / 0.004 of kappa(0).  Between EPs the peak half width ~ the narrow pole (0.25 vs
    0.23; 0.32 vs 0.27 ueV).  NONLOCAL G = T_ECT - T_CAR <= 0.018 e^2/h at all energies and points (largest between EPs):
    the transport stays neutral, G_nl shows no EP signature.  Deviations only beyond ~50 ueV (bulk levels) and next to the
    lower transition (565 ueV: bulk states at 10-100 ueV, charged).  No kink anywhere at an EP (cf. Li & Trauzettel Fig.
    2b).  Not done: tau 0.7 EP / outside (its resolved edges fell outside the selection window), the no-EP 'narrowest'
    picks sit at 565 ueV next to the lower transition (kappa(0) ~0.001, uninformative).
  - NOT DONE: finite-bias local / nonlocal conductance with asymmetric probes; where EPs occur relative to Q and the gapless
    strip; the L test averaged over all zeros (removes the lobe scatter); DOS(E) line shapes across an EP.
  - 2026-11-23: user judged the EP topic NOT thesis-relevant; Li & Trauzettel reproduction dropped.

## SCHARF: choosing REGIMES -- gate mu_C and probe bond tau (2026-11-24, `study_tuning.py` [regimes], plots/regimes_mixed.png)
* Plots only, from caches (study_mixed [deck] 4 Q / Delta_top on (mu_C 0-2.4, E_Z 0-2) at phi pi, tc 1; [leads] Gamma(tau);
  [tuning] coarse maps).  phi = pi, tc 1: Q = -1 on 68 % of (mu_C, E_Z), 56 % of that GAPLESS (all of mu_C >= 1.4).
  Proposed gate picks (Q = -1 / gapped E_Z, max Delta_top): R1 mu_C 0.3 (0.3-1.5 / 0.3-1.5, 48 ueV at 0.9: ends decoupled,
  short xi_M); R2 0.7 (0.1-1.3 / 0.1-1.3, 37 at 0.5: working point, ends overlap); R3 1.0 (0.1-1.3 / 0.1-0.9, 16 at 0.3:
  gapped only at low field); R4 1.6 (0.1-1.5 / none: Q = -1 but gapless, NO transport computed yet).  Bond picks (Gamma at
  mu_C 0.7): C1 0.2 (0.35 ueV, 4 mK, tunnel), C2 0.5 (3.2, 37 mK, matched), C3 0.7 (10, = 0.27 Delta_top, neutral share
  0.83 -> 0.14), C4 1 (37 ~ Delta_top, transparent).  Neutral share of kappa > 0.25: mu_C 0.3 0.91 / 0.84 (tau 0.3387 / 0.5)
  ~ mu_C 0.7; mu_C 1.2 0.65 / 0.60.  Gamma(tau) is measured at mu_C 0.7 only (Rashba mu_C 0.3 had ~3.5x larger Gamma at
  tau 0.3387) -> re-measure per gate pick.  Q / Delta_top over phi exist only at mu_C 0.7.  Picks NOT yet agreed with the user.
* GAMMA THREE WAYS AT ONE POINT (2026-11-24, scratch gamma_check.py, no study part; phi pi, mu_C 0.7, tc 1, eps zero E_Z
  0.285): local-Andreev HWHM ([leads] cache) / full pole -Im lambda (nh_spectrum) / first-order projection G_L (nh_pair_2x2):
      tau 0.2   0.340 / 0.339 / 0.317        tau 0.3387   1.099 / 1.097 / 0.907        tau 0.5   3.043 / 3.036 / 1.975
      (E_Z 0.345, tau 0.5: 3.311 / 3.302)    tau 1   37.2 / 52.06 / 7.90 ueV  (tau 0.7 pole not computed)
  HWHM = pole width to 0.1-0.3 % for tau <= 0.5 (one or both probes attached: same pole, the ends are decoupled at an eps
  zero; right Majorana width 0.001 ueV).  tau 1 FAILS: HWHM 29 % below the pole, bulk poles at +-26-51 ueV (widths up to
  7.6) lie inside the line -> no single-pole Gamma at tau 1.  First order is tau^2 EXACTLY (Sigma_L ~ tau^2 for one bond)
  and is 7 / 17 / 35 % low at tau 0.2 / 0.3387 / 0.5, x6.6 at tau 1 -> the measured Gamma ~ tau^2.6 is a beyond-first-order
  effect of the probe on the end mode (not the bare wavefunction).  Gamma at the operating points differs (one-probe
  pole at phi 1.7417 pi, eps crossings E_Z 0.84 / 1.03, [ep8]: tau 0.2 0.97, 0.4 4.72, 0.5 8.64 / 7.77 ueV -- ~2.8x the
  phi = pi, E_Z 0.285 values) -> use the pole per point.  [An earlier line said 4.72 for tau 0.5: wrong, that is tau 0.4.]
  WHY (scratch gamma_why.py, one probe, same point): Sigma_L = H_S - i Gamma_S / 2 on the contact column, max |H_S| 0.18 /
  0.51 / 1.11 / 2.18 / 4.46 meV and max Gamma_S / 2 0.03 / 0.09 / 0.21 / 0.40 / 0.82 meV for tau 0.2 / 0.3387 / 0.5 / 0.7 / 1
  (t = 10.03 meV; max |matrix element|; by EIGENVALUES max |H_S| / max (Gamma_S/2) = 1.96 for every tau, since
  Sigma_L ~ tau^2) -- the real part is ~2x the loss and >> Delta, mu_C.  Golden rule with the end mode RESHAPED by H_S
  (left Majorana of H_C + Sigma_rib + H_S, then <g|Gamma_S/2|g>) vs the exact pole: 0.080 / 0.081 (tau 0.1), 0.337 / 0.338,
  1.098 / 1.097, 3.073 / 3.036, 10.06 / 9.48, 68.0 / 52.1 ueV -> exact / reshaped 1.015 / 1.004 / 0.999 / 0.988 / 0.943 /
  0.766.  The end mode's weight on the contact column rises 1.88e-3 -> 2.26e-3 / 2.87e-3 / 4.71e-3 / 1.55e-2 (tau 0.3387 /
  0.5 / 0.7 / 1), the same factor as exact / first order (1.21 / 1.54 / 2.45).  => the probe's real self-energy pulls the
  end mode onto the contact; that, not Gamma vs the level spacing (3 vs 26 ueV), is why Gamma grows faster than tau^2.
  At tau 1 the loss itself is non-perturbative too (reshaped golden rule 31 % high).  tau 0.7: HWHM 9.633 vs pole 9.479
  (1.6 %).  tau 1: HWHM 37.2 vs pole 52.1 -- the line (52 ueV) overlaps the bulk poles at +-26-34 ueV (overlapping
  resonances), so no single Lorentzian.
  THE MATRIX i(Sigma_L - Sigma_L^dag) (scratch gamma_matrix.py): 4 open channels (e/h x spin), eigenvalue 0.370 / 2.314 /
  9.256 meV each at tau 0.2 / 0.5 / 1 (exactly tau^2; ~1000x the end-mode width).  Exact identity for the pencil A psi =
  lam M psi: -Im lam = <psi|Gamma_S|psi> / (2 <psi|M|psi>) with the EXACT right eigenvector -> 0.3370 + 0.0015 (ribbon eta
  loss) = 0.3385, 3.0356, 52.0559 ueV = the poles to every printed digit.  So golden rule / reshaped / pole are the same
  projection with the closed / H_S-reshaped / exact psi.  Factor 1/2: <psi|Gamma|psi> is the FWHM (Datta's level
  broadening); our Gamma (pole, HWHM) is half of it.
  PHYSICAL BARRIERS (scratch gamma_barr.py; one-site lam_barr, bulk bonds, folded into Sigma_L as [barr]; same point):
      lam (meV nm)   600    400    300    240    180    120    80     40     0 (= bond tau 1)
      |H_S|/(G_S/2)  14.9   10.6   8.4    7.1    5.8    4.5    3.7    2.8    1.96   (eigenvalue maxima)
      exact pole     0.224  0.498  0.874  1.345  2.313  4.769  9.149  22.56  52.06 ueV
      exact/first    1.33   1.50   1.68   1.86   2.16   2.74   3.53   5.33   6.59
      exact/reshp    1.006  1.003  1.001  0.999  0.996  0.989  0.981  1.000  0.766
  A barrier site of height V_b = lam / a, integrated out, leaves a real potential ~ -t^2 / V_b on the contact column but a
  loss ~ 1 / V_b^2 -> the bare golden rule is WORSE for barriers (33 % off already at Gamma 0.22 ueV), the H_S-dressed one
  BETTER (<= 1.9 % up to Gamma 22.6 ueV; bonds 5.7 % at 9.5 ueV).  Contact weight 2.5e-3 -> 9.5e-3 (closed 1.88e-3).
  EQUAL Gamma, cached [ep8] / [barr] lines (phi 1.7417 pi, E_Z 0.80-1.10, equal probes): |G|/kappa median (90 %) bond
  0.4 / lam 220 (Gamma_1 4.5 / 4.3): 0.012 (0.032) / 0.012 (0.030); bond 0.5 / lam 150 (8.3 / 8.7): 0.025 (0.072) / 0.026
  (0.064); bond 0.6 / lam 120 (14.6 / 12.7): 0.044 (0.12) / 0.036 (0.11); bond 0.8 (45.6): 0.20 (0.34).  -> charge grows
  with Gamma, not with the contact type: a barrier does not extend the usable coupling range.
  WHERE H_S COMES FROM (scratch sigma_closed.py, tau 1): Sigma_L projected on the probe's transverse modes n = 1..10 (ny 10,
  mu_N 3, t 10.03 meV) = the semi-infinite-chain closed form tau^2 t^2 g(x), x = mu_N - E_n - 2t (Capecelatro et al.
  arXiv:2412.01702 App. A Eqs. A4/A7/A8; their A7 = the level shift): n = 1 OPEN (E_1 0.812): -8.932 - 4.554i closed form vs
  -8.931 - 4.506i code; n = 2..10 CLOSED (evanescent, real only): -8.76 / -5.42 / ... / -1.84 vs -8.87 / -5.51 / ... / -1.85
  (<= 1.2 %); weak inter-mode coupling <= 0.76 meV not in the closed form.  Electron / hole blocks have opposite real
  parts (-U tau_z): the probe acts as a LOCAL CHEMICAL-POTENTIAL RISE of tau^2 x (1.8-8.9) meV on the contact column
  (one open channel near its subband bottom, Re/Im = 1.96, plus nine evanescent channels) -> the end mode is pulled onto
  the contact.  Broad-band at band centre would make Re Sigma = 0; the probe here is 2.2 meV above its subband bottom.
  Capecelatro et al. also give: quasi-bound states = poles of the retarded GF on the sub-gap sheet, G_pol = Z (z -
  H_eff)^-1 (Eq. 18; Z = weight matrix -- our pencil's M^-1 is its linearised, symmetrised form), polar vs branch-cut
  split (Eq. 16, C5), Fano line with complex residues (Eq. 23), validity: width << distance to other spectral weight
  (Eq. 25).  PHS (their C6) makes the residue of a pole on the imaginary axis REAL -> pure Lorentzian at an eps zero.
* GAMMA ALONG phi = pi (2026-11-24, `study_tuning.py` [gamma], plots/gamma_line_mixed.png, logs/study_tuning_gamma.log;
  cache scharf_absGM_poles_mixed_a10_pi).  Gamma = pole width (Capecelatro et al.) at every field of the [oscillation]
  transport line (E_Z 0.10-1.35 step 0.01), configurations (tau_L, tau_R) = (0,0), (t,0), (0,t), (t,t), t 0.3387 / 0.5;
  ~70 s per field (one ribbon set-up, 7 pencils), 6 workers ~25 min.  Closed pair |Re| = eps_box on 126 / 126 fields;
  Gamma_R = Gamma_L to 5e-10 (mirror).  Gamma(E_Z) over the pair range (E_Z <= 0.94): 0.55-1.48 / 1.56-4.05 ueV (old
  constant 1.15 / 3.18); drops above ~1.25 (transition).  eta: each mode carries eta0 = 0.00123 ueV from the ribbons ->
  subtract it ONCE from the one-probe pair sum (twice left eps_eff ~ sqrt(2 Gamma eta0), a 0.005 floor at low field).
  |kappa - 2x^2/(1+x^2)^2| on the neutral pair-range points (median / 90 % / max), tau 0.3387 | 0.5:
      x = eps_box / Gamma_const (old slides)     0.0124 / 0.074 / 0.168  |  0.0048 / 0.077 / 0.155
      x = eps_box / Gamma_pole(E_Z)              0.0031 / 0.035 / 0.177  |  0.0021 / 0.051 / 0.108
      x = eps_eff / Gamma_pole(E_Z) (poles)      0.0003 / 0.005 / 0.015  |  0.0006 / 0.011 / 0.024
  -> the per-field width removes the drift scatter; the remaining eps_box error is the probe's level shift near the eps
  crossings (eps_eff^2 = |l1 l2| - Gamma_L Gamma_R corrects it).  det r: NEGF vs 1 - 2 G_L G_R / |l1 l2| median 0.004 /
  0.003, max 0.03 / 0.06.  Fields with |l1 l2| < G_L G_R (eps_eff undefined): kappa = 0 at low field (eps << Gamma) and two
  CHARGED points at the eps zeros 0.79 / 0.92 (tau 0.5: kappa 0.036 / 0.163, G 0.035 / 0.141 = electron transport through
  bulk states, outside the two-pole form).  Slides redrawn: collapse_mixed, mixed_overlap, mixed_oscillation.
* REGIME CRITERIA (2026-11-24, `study_tuning.py` [classes], plots/classes_line_mixed.png; plots only, [oscillation] bulk
  modes + [gamma] poles; definitions and threshold sources in the part header).  Labels need NO transport:
  A topology (Q, Delta_top; gapless <= 0.01 ueV), B separation L/xi_M (merged < 1.5 <= overlapping < 4 <= separated),
  C x = eps_eff/Gamma (decoupled < sqrt2-1 <= resonant <= sqrt2+1 < split; core 0.82-1.22 = |det r| < 0.2),
  D Gamma/E_next (E_next = lowest non-pair closed-junction level; isolated < 1/3 <= marginal < 1 <= overlapping, PROVISIONAL).
  TEST on phi = pi, mu_C 0.7 (all Q = -1 gapped; separated 0.10-0.60, overlapping 0.61-0.91, merged 0.92-1.35 meV):
  B predicts CHARGE -- merged: neutral share 0.00-0.08 at both tau, also where kappa ~ 1/2 (merged + resonant: kappa median
  0.49 / 0.38, neutral 0.00 = the false-positive class); separated / overlapping: neutral 0.9-1.0.  C predicts kappa --
  resonant: kappa > 1/4 on 80-100 %; decoupled, split: 0 % (separated + decoupled: median kappa 0.006 / 0.001); core: 7 / 6
  fields, kappa 0.480-0.501 / 0.486-0.499.  D: all isolated here (Gamma/E_next <= 0.06 / 0.17) -> untested; needs tau 0.7-1
  or other gates.  A: only 'gapped' on this line -> trivial / gapless need other gates (R1-R4 of [regimes]).
* UNCORRELATED DISORDER, THESIS JUNCTION (2026-11-25, `study_tuning.py` [dis], plots/dis_points_mixed.png,
  dis_ldos_mixed.png, logs/study_tuning_dis.log; box W 0-8 meV = rms 0-2.3 meV, 16 seeds, tau 0.5 (+ tau 1), fixed E_Z;
  0 failed solves; tools._nh_setup fixed the same day to include disorder).  Medians (min-max) of kappa, and spectrum:
    sep (0.30, pi): kappa 0 up to W 8; strong-probe Andreev = 2 and det r_tau1 < 0 for ALL seeds up to W 4; E_next
      27.6 -> 23.0 (W 2) -> 13.9 (W 4) -> 2.9 ueV (W 8, gap filled: Q=-1 2/16, Andreev 0.1-1.8).  Robust to W 4.
    res (0.72, pi): kappa 0.499 / 0.495 / 0.481 / 0.455 / 0.295 / 0.157 at W 0 / 0.5 / 1 / 2 / 4 / 8, but the per-seed MAX
      stays 0.496-0.499 at every W (disorder DETUNES eps off Gamma, as in the old study); neutral 0.88-1.0 throughout;
      Andreev_tau1 ~2 falls 1.00 -> 0.44 (W 2) -> 0.12 (W 8).
    merged (1.06, pi): charged at every W <= 4 (neutral 0.00); kappa 0.523 -> 0.058 (W 4).
    triv (0.30, 0): kappa 0 at every W; strong-probe Andreev 1.7 clean, 0.4-2.5 with disorder, 1 / 16 seeds within 0.05
      of 2 at W 1 and at W 4 (a trivial zero-bias look-alike; local Andreev can exceed 2 here: two spin channels).
  LDOS (seed 0, E_Z fixed): sep keeps its end weight and zero-energy peak up to W 4, at W 8 the end weight collapses;
  res: end peaks shrink with W; triv: no zero-energy end state.  OPEN: per-seed E_Z sweeps (does every seed still reach
  kappa = 1/2 somewhere?).
* CORRELATED DISORDER, THESIS JUNCTION (2026-11-25, `study_tuning.py` [cdis], plots/cdis_field_mixed.png,
  cdis_points_mixed.png, cdis_ldos_mixed.png, logs/study_tuning_cdis*.log).  Gaussian random field
  <V V'> = sigma^2 exp(-r^2 / 2 lambda^2) (junction.disorder_potential, dist='gauss'; field statistics checked over
  100 seeds, box path bit-identical to [dis]); rms sigma 0.29 / 0.58 / 1.15 meV (= box W 1 / 2 / 4), lambda 0 / 10 /
  30 / 100 / 300 nm, 16 seeds, sep / res / triv points as [dis], 0 failed solves.  lambda 0 reproduces box at equal rms.
    sep: Q = -1 (tau 1) and Andreev ~2 in all seeds only while sigma * lambda is small: lost at sigma 0.58 for
      lambda >= 30 (Q=-1 1/16, 2/16, 6/16 at 30 / 100 / 300) and at sigma 1.15 already for lambda 10 (1/16); box keeps it
      to rms 2.3.  Mechanism from the poles: the low pair's probe coupling sqrt(G_L G_R) drops ~100x (2.9 -> 0.01-0.06 ueV)
      while its splitting rises only 0.04 -> 1-2.7 ueV (E_next 4-14 ueV): the zero modes leave the ends.  NOT a global
      mu_C offset: at lambda 30 the junction-averaged V has std 0.06 meV; only at lambda >= 100 the surviving seeds are
      those with a mu-raising average shift (-0.2 to -1.5 meV vs +0.1 for the lost ones).  LDOS (sigma 1.15, seed 0):
      white keeps the E = 0 end peaks, lambda 10 / 30 lose them, lambda 100 shows an INTERIOR E = 0 peak near 1.05 um
      (consistent with a domain wall; one seed only).
    Estimate (not computed in the model): Born backscattering ~ S(2 k_F) = 2 pi sigma^2 lambda^2 exp(-2 k_F^2 lambda^2);
      k_F a ~ sqrt(mu_C / t) = 0.26 (lambda_F ~ 240 nm) gives S / S_white ~ 5 (lambda 10), ~17 (lambda 30), then
      exponentially small (lambda >= 100), where the damage is the smooth local-mu landscape instead.  Equal rms is
      therefore NOT equal disorder; compare at equal mean free path (pending mfp calibration).
    res: per-seed max kappa still 0.497-0.499 up to sigma 0.58 at every lambda, median 0.47 -> 0.18-0.25 (lambda >= 30);
      at sigma 1.15, lambda >= 30 the coupling collapses as at sep and kappa -> 0.  One seed (sigma 1.15, lambda 300,
      seed 11) has kappa 0.70 > 1/2 with |G|/kappa ~ 0.5: two low pairs (0.46, 3.3 ueV), multi-level, NOT checked further.
    triv: kappa 0 and Q = +1 for every sigma, lambda and seed: no false topological signal; tau-1 Andreev scatters 0-3
      (trivial Andreev states at a transparent contact; Andreev ~2 is no discriminator at tau 1, det r is).
  Rendered concisely by [disfig] (plots/disfig_box_strips.png, disfig_corr_strips.png, disfig_end_weight.png; seed 0 only).
* GAP DESIGN (2026-11-26, `study_tuning.py` [gapdesign], plots/gapdesign_mixed.png, logs/study_tuning_gapdesign.log):
  bulk Delta_top at phi = pi (tc 1, Delta 0.25, a 10) over W 50-200 x SOC x mu_C 0.1/0.3/0.7 x E_Z 0.05-2.
  Max over ALL E_Z: W barely matters for 50-100 nm (falls beyond); Rashba 32 at mu_C 0.1 141 ueV (W 80), mixed 16 51
  (W 100, mu_C 0.1) vs 37.5 at the thesis point; Dresselhaus admixture ~halves the gap, doubling Rashba ~doubles it.
  BUT those maxima sit at E_Z 1-1.8 meV (B = 3.45 E_Z[meV] T at g 10, i.e. 3.5-6 T) beyond a thin Al film's in-plane
  critical field (~2-3 T; NOT modelled: Delta is field-independent here).  Capped at E_Z <= 0.6 (2.1 T): mixed 16 best
  37.5 (mu_C 0.7, W 100) / 40.4 (mu_C 0.3, W 150); mu_C 0.1 at W 100 only 16 -- DEPLETING THE CENTRE DOES NOT HELP at
  realistic fields; Rashba 32 108 (mu_C 0.3, W 100).  Cap 0.8 (2.8 T): mixed 16 42.9 (mu_C 0.3, W 100), Rashba 32 122.
  W 50 maxima sit at the E_Z grid edge (topological beyond E_Z 2): underestimated, but unreachable fields anyway.
* DEPLETED CENTRE (2026-11-26, [depl], plots/depl_transport_mixed.png, depl_spectrum_mixed.png; field cap shaded at
  E_Z 0.6 / 0.8): thesis junction at mu_C 0.1 / 0.3 / 0.7, phi = pi, E_Z 0.05-2, tau 0.5 (+ tau 1), 0 failed solves.
  Q = -1 (tau 1) from E_Z 0.40 / 0.25 / 0.05.  Depletion gives a clean strong-probe Andreev = 2 over E_Z ~0.45-1.1
  (mu_C 0.1) / 0.35-1.0 (0.3) and a ~3x smaller median splitting eps (0.6 / 0.45 vs 1.75 ueV), but kappa(tau 0.5) has
  NO clean 1/2 plateau at any mu_C: depleted kappa only reaches ~0.45-0.5 at E_Z 1.1-1.5 (> 3.8 T); the thesis point
  touches 0.49 at E_Z 0.75.  Gamma(tau 0.5) ~3.7 ueV at all mu_C (probe-set).  Why kappa stays low while Andreev = 2:
  NOT checked.
* FINE CORRELATED DISORDER (2026-11-26, [dfine], plots/dfine_map_mixed.png, dfine_collapse_mixed.png): sep point,
  sigma 0.1-1.2 meV x lambda white/10/20/30/50/100/200 nm x 16 seeds, 0 failed solves.  P(Q = -1, tau 1) = 1/2 at Born
  l_tr ~ 250 (lambda 10), 220 (20, 30), 280 nm (50): short-range disorder COLLAPSES onto l_tr ~ 200-300 nm (~ xi_M).
  White noise never fails (l_tr >= 740 nm on the grid).  lambda 100 fails at l_tr ~ 750 nm, lambda 200 at ~1200 nm and
  P creeps down (0.2-0.4 even at l_tr 200-500): smooth landscapes (lambda >~ lambda_F / 2) are not Born scattering, they
  act as a seed-dependent local mu_C -> l_tr is the wrong yardstick there.  Andreev_tau0.5 ~ 2 fails at ~ 1.5x smaller
  sigma than Q.  sigma scale at the sep point: survives sigma <= 0.3 meV for any lambda; lost by 0.5-0.6 for 20-100 nm.
  Advisor versions (one message each, same cache): plots/dfine_adv1_strength.png, dfine_adv2_meanfreepath.png,
  dfine_adv3_map.png.

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
[HISTORICAL: 2026-10-08/12, OLD bonds, a 20, alpha 14.3 / beta 7.3, made with disorder.py (deleted 2026-11-25).  The
thesis-junction disorder study is study_tuning [dis] (box) and [cdis] (Gaussian-correlated); see "UNCORRELATED
DISORDER, THESIS JUNCTION" under [regimes].]
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
* REVISED 2026-10-12 with the scattering invariant Q = sign det r (left lead, tc_barr=1;
  `_det_r` in `computations/disorder.py`; clean A -> -1.00, B -> +1.00).  The bullets above
  hold up to W ~ 3.2 meV, but at STRONG disorder two of them break:
  - "trivial" is no longer a property of the point.  At B with W=6.4 meV, 9/256 realisations are
    TOPOLOGICAL (Q=-1); the kappa=0.478, |G|/kappa=0.006 seed with LDOS peaks at both ends is one of
    them -- a disorder-induced topological state, NOT a fake.
  - but genuine fakes exist too: TRIVIAL (Q=+1, |det r| 0.84-0.99) realisations with NEUTRAL kappa
    up to 0.38 (B W=6.4 seed 233: kappa 0.380, |G|/kappa 0.018, LDOS candidate; A W=6.4 seed 50:
    0.333, det +0.967).  First seen at W=4.8 meV (19 Delta).  So "trivial look-alikes are
    transport-dark" holds only for W <~ 3.2 meV.  No trivial realisation reached kappa > 0.45, but
    nobody has swept E_Z per seed at W >= 4.8 -- whether one can be TUNED to 1/2 is open.
  - A itself loses topology between 3.2 (62/64 Q=-1) and 4.8 meV (21/64).
  - "2e^2/h is not sufficient" is now MEASURED: 1.5-5% of trivial realisations at B/C
    (W=1.6-3.2) have the strong-probe Andreev conductance within 0.05 of 2.
### Weak disorder, W <= Delta = 0.25 meV  (2026-10-12; mean free path >> L, ballistic)
* Point A, 64 seeds: mean kappa 0.4945 -> 0.4868 at W=Delta, worst seed 0.435, 97% above 0.45;
  |G|/kappa <= 4.4e-4; strong-probe Andreev >= 1.993; Q=-1 and LDOS candidate in 64/64.
  Essentially undamaged.
* B and C, 128 seeds at W=0.1/0.25: no LDOS candidates, no Q=-1, nothing near quantised Andreev.
  B kappa <= 3e-10.  C kappa up to 0.028 but CHARGED (|G|/kappa 0.56-0.79; clean C is 0.001 with
  G~kappa: the gap nearly closes at E_Z=0, phi=pi) -- rejected by the G~0 prong.
* E_Z sweep at phi=pi (0.005 meV steps, 16 seeds): every seed reproduces the clean resonance
  PATTERN.  The broad resonances (0.505-0.64 meV) keep kappa 0.44-0.49 and move <= 0.02 meV at
  W=Delta.  The dense low-field ones (E_Z < 0.13, spacing ~0.025) move by up to a full spacing, so
  kappa AT the clean position can drop to 0.002 while each device still has its own peak at
  ~0.45-0.48: the resonance is device-specific, its height is not.  The peaks at 0.845 and 0.98
  are narrower than the grid; their heights are not resolved at any W.
* So for realistic (weak) disorder, the two-pronged test is untouched; disorder only matters
  above ~6 Delta, and fakes only above ~19 Delta.
* Code: was `computations/disorder.py` (deleted 2026-11-25); its det r lives on as tools.det_r.  Every number in this
  section was measured at A = (0.505, pi) with the old bonds.

## Measurability — can an experimentalist see this?  (conversion: 1 ueV = 11.6 mK)
### SCHARF: no (2026-10-08) -- SUPERSEDED IN PART: read to the end of this subsection.
* [The first four bullets are 2026-10-08: OLD bonds, one fixed E_Z per case, alpha 14.3 / beta 7.3.  From 10-21 on
  (new bonds, full field scans): neutral kappa ~ 0.48-0.50 at 20 mK and 0.35 at 50 mK at the coupling optimum
  tc_barr 0.35-0.4 (pure Rashba W 100); the mixed SOC ~10-20 % lower.  So: marginally visible at 10-20 mK, as a
  resonance whose window is Gamma -- not 'no'.]
* The genuine Majorana resonance needs a WEAK probe (tc_barr~0.2) and is then ~0.5-0.8 ueV wide:
  kappa = 0.491 -> 0.44 at kT=0.1 ueV (1 mK) -> 0.019 at 1 ueV.  A fridge delivers 10-20 mK.
* Raising tc_barr to 0.5 widens things and the resonance MOVES in E_Z (to 0.79-0.85 meV) rather
  than vanishing -- so the earlier "strong coupling kills it" was measured at fixed E_Z.  But
  retuning does not rescue it: at L=800/1200 nm the wide kappa~0.5 peaks are CHARGED
  (|G|/kappa = 0.62 / 0.28), i.e. the false-positive class.  Only L=2000 nm stays roughly neutral
  (|G|/kappa=0.060) and there kappa(phi=0)=0.335, so the phase contrast is only 0.16 and kappa
  still halves by kT=2 ueV (23 mK).
* Every configuration is EITHER Majorana-like with no temperature window OR wide but charged.  [SUPERSEDED: the
  tc 0.35-0.4 window below is neutral and survives 20 mK.]
* PURE-RASHBA CASES SURVIVE ~10x HOTTER than the numbers above (different SOC, mu and field
  protocol -- the cause is NOT isolated) (2026-10-21, `study_rashba.py` [T] cell 1, a = 20, tc_barr 0.2, best kappa over a
  small E_Z window around each T = 0 neutral peak; checked against myf.linear_response to 0.4%):
      T [mK]                0      2      5      10     20     50
      W 100, L 2 (1 ABS)    0.499  0.498  0.48   0.36-0.37  0.16-0.17  0.03
      W 100, D 0.5 (2 ABS)  0.498  0.498  0.49   0.40-0.42  0.20-0.23  0.04
      W 300, L 4 (2 ABS)    0.50   0.48-0.49  0.32-0.35  0.13-0.16  0.04  0.02
  Half height at ~13 mK (W 100), ~15 mK (Delta 0.5), ~5-6 mK (W 300, L 4): the wider junction needs
  a longer L, hence smaller delta = Gamma, hence a narrower resonance.  |G| stays <= 0.02 at all T:
  heating lowers the peak, it does not charge it.  Contrast caveat: at W 100, E_Z ~ 0.7-0.9 meV,
  phi = 0 is ALSO topological (Q(0) = -1, the lobe centred on phi = 0 opens at E_Z ~ E_T) and its
  kappa(0) = 0.07-0.49 peaks are neutral (|G|/kappa 0.001-0.02): Majorana too, so kappa(pi) - kappa(0)
  is NOT a topological contrast there.  W 300, L 4 (E_Z < E_T): Q(0) = +1, kappa(0) ~ 0 -- genuine.
  NB the window search above UNDERSTATES tc 0.2: a full fine E_Z scan gives 0.43 / 0.32 at 10 / 20 mK.
* THE COUPLING HAS A TEMPERATURE OPTIMUM (2026-10-21, `study_rashba.py` [T] cell 2, W 100, L 2, phi = pi,
  E_Z 0.55-1.25 in 0.0025 steps; best kappa_T with |G_T|/kappa_T < 0.05 / < 0.10):
      tc     0 mK          10 mK         20 mK         50 mK
      0.2    0.499         0.432         0.317         0.174
      0.3    0.499         0.492         0.439         0.299
      0.35   0.500         0.500         0.469         0.336
      0.4    0.500         0.503         0.484         0.349 / 0.398   <- optimum
      0.45   0.23 / 0.50   0.22 / 0.50   0.18 / 0.48   0.00 / 0.43
      0.5    0.15 / 0.46   0.15 / 0.46   0.00 / 0.43   0.00 / 0.31  (|G|/kappa ~ 0.07, E_Z 0.905)
      0.7    0 / 0  (|G|/kappa ~ 0.4);   1.0: |G|/kappa ~ 0.8
  Larger Gamma widens the resonance (kappa barely moves with T at tc >= 0.5) but charges it as Gamma
  approaches E_1: the neutral window is tc ~ 0.35-0.4, and the E_Z range with neutral kappa > 0.25
  shrinks as tc grows (0.15 meV at 0.3 -> 0.05 at 0.4, 20 mK).  Energy grid stops at 35 ueV: fine to
  50 mK (~2% low), wrong beyond.
  W 100, Delta 0.5 (2 ABS), E_Z 0.3-1.0, tc 0.2 only (0.3-0.5 stopped, not run): 0.498 / 0.450 /
  0.348 / 0.200 at 0 / 10 / 20 / 50 mK, all |G|/kappa <= 0.01 -- ~10% above Delta 0.25 at the same tc,
  neutral window 0.24 vs 0.13 meV at 10 mK.  Whether its larger E_1 tolerates tc > 0.4 is OPEN
  (add the couplings to SCAN_CASES in study_rashba.py [T]; the finished lines are cached in plot_npz/scharf_absTtc_*).
* MORE KNOBS AT FINITE T (2026-10-23, study_rashba.py [T] cell 2; best NEUTRAL kappa, |G|/kappa < 0.05,
  over the whole field scan; 0 / 10 / 20 / 50 mK):
      W 100 mixed SOC   tc 0.2: 0.498/0.375/0.250/0.126   0.3: 0.499/0.472/0.391/0.244   0.4: 0.500/0.499/0.454/0.06
      W 100 mu_N 1.0    tc 0.3: 0.497/0.443/0.331/0.187   0.4: 0.499/0.489/0.430/0.285
      W 100 mu_N 2.0    tc 0.3: 0.499/0.484/0.415/0.272   0.4: 0.500/0.501/0.477/0.343
      W 100 mu_N 3.0    tc 0.3: 0.499/0.492/0.439/0.299   0.4: 0.500/0.503/0.484/0.349   (default)
      W 100 mu_N 5.5    tc 0.3: 0.499/0.494/0.450/0.316   0.4: 0.500/0.503/0.487/0.352
      W 300 L 4         tc 0.2: 0.499/0.335/0.215/0.151   0.3: 0.501/0.451/0.354/0.207   0.4: 0.501/0.480/0.314/0.258
  - Mixed SOC is 10-20% lower at 20 mK (smaller Delta_top) and has CHARGED kappa ~ 0.57 peaks
    (|G|/kappa 0.5-0.6) at every tc.  It is NOT why these numbers beat the old Measurability ones:
    the full field scan and the coupling are.
  - mu_N (probe lead) acts as another Gamma knob: more open lead subbands (5-site strip: 0.67 / 2.49 /
    4.98 meV) = larger lead DOS = larger Gamma at fixed tc = slightly more T-robust.  A single-channel
    lead (mu_N 1-2) does NOT remove the charged peaks (best overall kappa still |G|/kappa ~ 0.27-0.3).
  - W 300, L 4 also has its optimum at tc ~ 0.3-0.4 but stays ~25% below W 100 at 20 mK (narrower
    resonances); its 50 mK maxima sit at the window edge E_Z = 0.1 meV -- not reliable.
* The W lever for Scharf (2026-10-08: untested -- since TESTED, see the end of this bullet) is NOT the Gresta ny lever: here W sets the Thouless
  energy, E_T = (pi/2)*2*sqrt(mu_N t)/n_across ~ 1/W, and the topological region sits at
  E_Z ~ E_T, so a WIDER junction needs a WEAKER field (W=100 nm -> 2.9 T, W=200 -> 1.4 T,
  W=300 -> 1.0 T).  Opposite direction to Gresta.  Whether that trades field for temperature
  favourably is open.  [ANSWERED 10-21/23: no -- W 300 needs L ~ 4 um and is LESS T-robust (half height 5-6 vs 13 mK
  at W 100; ~25 % below W 100 at 20 mK even at its optimal tc).]

### THE PLATEAU THAT DOES EXIST IN SCHARF: local Andreev at 2e^2/h
There is no kappa plateau (see above), but the LOCAL Andreev conductance has one, because it
needs only ONE Majorana at the probed end -- no overlap, hence no delta, hence no resonance.
Measured at phi=pi, L=2000 nm, scanning E_Z in 0.01 meV steps:
      tc_barr   longest flat run        mean Andreev   deviation from 2
        0.2     0.18 meV (0.18-0.35)      1.9357           3.2%
        0.5     0.51 meV (0.05-0.55)      1.9723           1.4%
        1.0     0.74 meV (0.05-0.78)      1.9769           1.2%
  At tc_barr=1.0 it reads 1.999 to FOUR digits across E_Z = 0.11-0.53 meV.  kappa over the same
  range is 0.000-0.001: the knob that builds the plateau destroys the nonlocal signal.
* Its UPPER EDGE tracks the topological transition: 0.78 meV at tc_barr=1.0 against E_T=0.829.
  So plotting the plateau maps the phase boundary.
* TEMPERATURE-ROBUST, because its width is ~Gamma (your choice) and not ~delta.  At tc_barr=1.0,
  E_Z=0.3: energy half-width 20.6 ueV (~239 mK), and zero-bias dI/dV =
      1.9840 at 11.6 mK   1.7514 at 58 mK    1.1641 at 232 mK
      1.9429 at 23.2 mK   1.4672 at 116 mK   0.8013 at 580 mK
  At tc_barr=0.2 (what nonlocal kappa needs) the peak is only 0.47 ueV wide and has already
  collapsed to 0.72 by 11.6 mK.
* NUMERICAL CAVEAT: `conductances(...)` at kT=0 returns ~50 here -- a finite-difference artefact
  (dV=1e-5 meV against a 1.25 ueV energy grid).  Use the channel value eh_local+he_local at E=0,
  or a finite kT.

### WHY THERE IS NO kappa PLATEAU -- the structural reason
* The end states DO overlap enough to carry heat: kappa=0.4910 with G=0.0001 at L=2000 nm, the
  full half-quantum.  The failure is WIDTH, not amplitude (0.5 ueV in energy, 0.03 meV in E_Z).
* Widening needs a larger splitting delta so a larger Gamma can be matched -- but delta IS the
  overlap, and increasing it merges the two states into an ordinary Andreev level.  Measured:
  L=800 nm at tc_barr=0.5 gives kappa=0.5518 but G=0.343 (|G|/kappa=0.62), i.e. charged.
* So: MAJORANA CHARACTER REQUIRES SEPARATION; MEASURABLE NONLOCAL HEAT REQUIRES OVERLAP.  The
  better the Majoranas, the narrower the heat signal, and no window satisfies both.  This is
  definitional, not an engineering shortfall, which is why tuning L, tc_barr, E_Z or phi all fail.
  [LATER (2026-11): extended eps ~ Gamma regions DO exist along phi (mixed SOC best: 1.37 pi at one field, but only
  50 ueV wide in field; Rashba 48 maps 9-11 % of (E_Z, phi), not yet restricted to Q = -1).  They are this same
  resonance held over a range, not protected -- the conclusion 'no PROTECTED kappa plateau' stands.]
* THE RESULTING TENSION, both halves measured in this model:
      the quantity WITH a plateau (local 2e^2/h) CANNOT prove a Majorana -- disorder fakes it,
          with BETTER LDOS contrast (17-19) than the genuine MBS (3.2-6.4), at kappa ~ 1e-7;
      the quantity that CAN prove one (nonlocal kappa=1/2 with G=0) has NO plateau.
* RESOLVED (mu_N scanned; mu_N = today's mu_C, the centre): kappa oscillates in mu_N exactly as it does in E_Z, with quantised
  maxima and no plateau; the Andreev plateau DOES extend across mu_N, making it a 2D wedge.
  See "The plateau is TWO-DIMENSIONAL in (E_Z, mu_N)" below.

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

### The plateau is TWO-DIMENSIONAL in (E_Z, mu_N) -- mu_N scanned 2026-10-08.
[OLD bonds; one mu for leads AND centre then -- 'mu_N' below = today's mu_C, but every change moved the probe leads too.]
* mu_N was the last untested parameter, and it CONFIRMS the plateau rather than shrinking it.
  At phi=pi, tc_barr=1.0, the local Andreev conductance holds 2e^2/h over a connected WEDGE:
  86 cells of a 29x31 map, mu_N = 0.65-0.95 meV x E_Z = 0.04-0.72 meV, G = 1.952-1.999
  (mean 1.9899, std 0.013).  The largest fully-inside RECTANGLE is
      mu_N = 0.65-0.90 meV  (0.25 meV)   x   E_Z = 0.08-0.40 meV  (0.32 meV),
  mean 1.9949 throughout.  The working point mu = 0.7 (today's mu_C) sits inside it.  mu_N is GATE-tunable, so this is a
  far stronger experimental claim than the 1D E_Z line: the plateau can be found by sweeping a
  gate at fixed field, and it does not require hitting a field value.
  Cached: `computations/plot_npz/scharf_mu_{map,line}.npz` (made by archive/mu_scan.py; the figure scharf_mu_map.png
  no longer exists).
* THE WEDGE CLOSES AS E_Z RISES: the mu_N width goes 0.30 meV (E_Z=0.12-0.32) -> 0.25 (0.36-0.40)
  -> 0.20 (0.44-0.48) -> 0.15 (0.52-0.56) -> 0.10 (0.60) -> 0 by E_Z=0.68.  So LOW field is
  better, which is the opposite of the usual "push the field up to get topological" instinct.
  I predicted the upper edge would GROW like sqrt(E_Z^2 - Delta^2) (a 1D-wire topological
  criterion); it does the reverse.  Unexplained -- do not quote a mechanism for it.
* THE LOWER EDGE IS THE FIRST TRANSVERSE SUBBAND, and it is E_Z-INDEPENDENT.  For a ny-site
  strip the subband bottoms are E_n = 2t(1-cos(n pi/(ny+1))); at ny=5, t=2.4877 meV that is
  0.667, 2.488, 4.975, ... meV.  Everything (kappa, G, Andreev) is EXACTLY ZERO below
  mu_N ~ 0.62 because the normal strip has no propagating mode at all.  [CORRECTED 10-14: it is the PROBE LEADS
  closing (the same mu set them), not the centre strip -- see 'probe leads, mu_N vs mu_C'.]  The measured edge sits
  at 0.62 against the bare 0.667, i.e. Zeeman/SOC soften it by only ~0.05 meV, NOT by the
  E_Z/2 = 0.15 I expected, and the edge does not move with E_Z across the whole map.
* THE SECOND SUBBAND IS AT 2.49 meV, far outside any usable window: the entire Scharf result
  lives on a SINGLE transverse subband.  Worth stating in the thesis -- it is why W enters only
  through E_T and why nothing is gained by adding transverse modes here (contrast the Gresta
  case, where ny=8 vs 12 matters a lot).
* kappa HAS NO PLATEAU IN mu_N EITHER, which closes the question.  On the 0.01-meV line scan at
  E_Z=0.3, kappa swings between 0.0005 and 0.4969 between adjacent points; on the map the longest
  contiguous run of |kappa-0.5|<0.05 along mu_N is 3 cells (0.10 meV) and that is at grid
  resolution.  Maxima are quantised (max among charge-neutral points = 0.4972) -- oscillation
  with quantised peaks, exactly as in E_Z.  Both tuning parameters now show the same thing, so
  "kappa oscillates, it does not plateau" is a measured statement about the model and not a
  failure to find the right knob.
* |G|/kappa SEPARATES THE WEDGE FROM EVERYTHING ABOVE IT.  Along the E_Z=0.3 line |G|/kappa stays
  below 0.015 for mu_N <= 0.94 and then climbs monotonically 0.10 (mu_N=1.00) -> 0.41 (1.20) ->
  0.87 (1.50).  The Andreev plateau and charge neutrality die together at mu_N ~ 0.95-1.0: above
  it the zero-energy state is an ordinary charged Andreev level.  Same discriminator as on the
  (E_Z, phi) map, now validated on a second cut.
* CAUTION: eh_local + he_local EXCEEDS 2 (2.12-2.21) at mu_N = 0.95-1.00, E_Z >= 1.16.  That is
  a separate high-field feature outside the wedge, not a plateau; the 2e^2/h bound only holds for
  a single-mode probe, so do not read those cells as "better than quantised".

### How an experimentalist would actually identify a Majorana (from this work)
1. A 2e^2/h zero-bias peak is NOT sufficient: disorder produced zero-energy end states with
   BETTER LDOS contrast (17-19) than the genuine MBS (3.2-6.4), carrying kappa ~ 1e-7.
2. The decisive measurement is NONLOCAL -- does heat cross from one end to the other.  That needs
   a delocalised PAIR, which is what being Majoranas means, and disorder cannot fake it -- up to
   W ~ 3.2 meV (13 Delta).  At W >= 4.8 meV trivial realisations reach neutral kappa ~ 0.3-0.38
   (see "REVISED" under Disorder), so the claim needs that bound.
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
* RECHECK 2026-11-11 (`study_mixed.py` [combine], logs/study_mixed_combine.log, logs/combine_dresselhaus_kT.log,
  plots/combine_checks.png; no script of the 10-07 runs survives; working point of archive/early_tests/01_single_point.py).
  The 10-07 transport numbers above are at kT = 1e-3, eta = 1e-5 (NOT stated above) -- there they reproduce:
  Bxy_c 0.1, theta_z 0: kappa(pi) 0.1475, kappa(0) 0.6518; Bxy_c 0.8 (kT 0): 0.4461, G 0.4085; '100' beta 0.4/0.6/0.8/1.1:
  kappa(pi) 0.909/0.889/0.986/0.864, G 0.32-0.37; the nx bullet is '110': beta 0.8 nx 80 -> 240: 0.3575 -> 0.1923.
  Corrections: (a) GRESTA working point "kappa = 0.497 at kT 1e-3, eta 1e-5": 0.4744 at eta 1e-5, 0.4969 at eta 1e-6.
  (b) in-plane only: refined min |E| 1e-19..1e-17 (exact nodes; the 1e-4 was the grid); chern_number returns 0 OR -1
  depending on grid N and field direction (meaningless either way); with Bz_s 0.94 + Bxy_s 0.94 it returns -1.
  (c) tilt: gap 0.2665/0.1909/0.1147/0.0382/0 at Bxy_s 0/0.1/0.2/0.3/0.4 (closed by 0.4), both in-plane directions.
  (d) '110' beta 0.8 at eta 1e-6, kT 0: NaN at nx 80 (failed lead solve, counted), 0.993/1.004 at nx 160/240 as in
  Numerics, but 0 overflow warnings now.  Also: kT 0 values differ (theta 0, Bxy_c 0.1: 0.430/0.436; '100' kappa(pi)
  1.07-1.59 for beta >= 0.4); at Bxy_c 0.05 (theta 0) the contrast survives (0.4215 / 0.0000, G 3e-4); theta pi/2 makes
  every Bxy_c > 0 worse (kappa 0.70-3.1, charged).  Populated centre mu_c 0.1/0.3: kappa(pi) 1.90/2.31 with G 1.1.
  Chern sign at beta 0: -1 for '110', +1 for '100' (different SOC forms in the code), identical transport.
* SAME QUESTIONS AT THE THESIS PARAMETERS (2026-11-11, [combine] B1-B3; single points, not tuned):
  - B1 proximitized ribbon (mu_S 1, Delta 0.25, m* 0.038, mixed 16 meV nm), out-of-plane Vz in the ribbon: Chern 0
    for Vz <= 0.9 meV, +1 for Vz >= 1.00 (N 200 = 400; a 10 = a 20); gap at 1.00 = 31.0 ueV for mixed AND Rashba
    (the k = 0 gap: transition at ~0.969 = sqrt(Delta^2 + (mu_S - E_so)^2), derived).  Largest bulk gap in the Chern
    phase: mixed 70.5 ueV (a 10, Vz 1.05) -> 35 at 3 meV; pure Rashba 16 132 ueV (Vz 1.2).  Vz 1 meV = 3.5 T at g 10.
  - B2 junction a 20 with Chern ribbons (Vz 1.00 / 1.05 in the ribbons only), transparent probe tau 1, DEPLETED centre
    mu_C -0.2: kappa(pi) 0.45-0.50 and kappa(0) 0.47-0.50 at in-plane E_Z 0 / 0.3 / 0.93, |G| <= 0.002 (one outlier
    kappa(pi) 0.682, Vz 1.05, E_Z 0) -- neutral ~1/2 at BOTH phases, no phase contrast.  Populated mu_C 0.7: kappa(pi)
    1.01-1.02 at E_Z 0, 0.39-0.43 at 0.3/0.93, |G| <= 0.04.  Weak probe tau 0.2: kappa <= 0.09 (one kappa(0) 0.481).
    Control Vz 0: 0 (depleted) / Scharf-like (populated).  No failed lead solves.  A plateau in parameters NOT tested.
  - B3 out-of-plane Bz in the normal strip only (thesis junction a 10, tau 0.5, in-plane E_Z 0.93; Bz_n = 0): Q = -1 up
    to Bz 0.4 meV at pi and 1.742 pi; Delta_top 18.98 -> 13.48 (pi), 48.7 -> 36.7 ueV (1.742 pi); xi_M 1394 -> 2016,
    669 -> 817 nm; kappa at fixed E_Z: pi 0.469/0.493/0.491/0.246/0.076, 1.742 pi 0.482/0.483/0.486/0.490/0.064 for
    Bz 0/0.05/0.1/0.2/0.4 meV (eps shifts; not re-tuned).
