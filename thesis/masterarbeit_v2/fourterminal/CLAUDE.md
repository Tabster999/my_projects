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

## FORMULAS: see DERIVATIONS.md (2026-11-06)
Every computed quantity (model, leads, RGF, channels, kappa, G, Andreev, det r, two-Majorana relations, Gamma, Q,
xi_M, eps, overlap law, end modes, LDOS): equation, derivation, physics, verified literature source (equation
numbers checked against the arXiv LaTeX/PDF) and the code lines.  Keep it in sync when a formula changes, and when
code lines move (the links name line numbers).  The .md is the source: after editing it, rebuild DERIVATIONS.pdf with
`py derivations_pdf.py` (pandoc + lualatex; the PDF's code links are vscode:// links, its § and arXiv IDs are
clickable), and keep display equations within the A4 width (break long ones with aligned/gathered).  Found while
writing it: Gresta et al. Eq. (4) as printed lacks the 3/pi^2 of its own T -> 0 limit (the code uses the normalised
kernel); the studies evaluate channels R -> L ('left'), which equals L -> R for kappa at E = 0 (unitarity + PHS).

## CODE LAYOUT of computations/ (merged 2026-11-06 -- keep it small)
| file | parts (former scripts) |
|---|---|
| `tools.py`, `ldos.py` | shared functions; PhysParams |
| `study_rashba.py` | W, delta, tc, checks, T, transition, soc_shift, fermi (check_fermi), a -- pure Rashba, mostly a = 20 |
| `study_oscillation.py` | oscillation, branches, fingerprint -- eps(E_Z), pure Rashba a = 20 |
| `study_tuning.py` | topology, coupling, tuning -- a = 10 couplings / topology / survey maps |
| `study_mixed.py` | oscillation, plateau, amplitude -- the mixed SOC (thesis case) |
| `slides_nov.py`, `figures.py`, `disorder.py` | advisor-deck figures, Scharf paper figures (own parameters), disorder -- CAUTION: disorder.py takes ldos.PP, now an a = 10 sandbox, NOT its documented junction |
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
  oscillation_ / plateau_ / amplitude_mixed.log line for line (every line of logs/study_oscillation, study_T, study_a,
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
  failed solves seen, but untested territory.
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
  (two-Majorana model scan, symmetric is always optimal at finite T too).
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
  0.6 / 0.8 / 1.0 / 1.2 / 1.35.  xi_hyp/xi_M 0.85-1.3.
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
  sets the temperature scale (finite T not computed).
* phi = pi at 0.930 is NOT part of it: kappa 0.469 but G 0.107 (0.23 kappa), box eps 0.41 ueV vs det-r x 0.76 ->
  not two-Majorana (box zero 0.93; the NEGF det r minimum sits ~0.01 meV lower).

## SCHARF: kappa peaks and the sign of det r -- bulk transition or eps = Gamma? (2026-11-04, study_tuning cell 8)
* Mixed SOC, mu_C 0.7, L 1/1.5/2/3 um x tau 0.3387/0.5/0.7 coarse maps; bulk Q on the same grid
  (scharf_absTune_Q_mixed_mu0.7, 45 % of the map Q = -1).  Plots detr_kappa_mixed (kappa, white det r = 0,
  red dashed Q boundary), detr_crossings_mixed.
* det r = 0 => r singular => one fully transmitted eigenchannel => kappa >= 1/2.  27 det r = 0 crossings refined by
  regula falsi (5 NEGF solves each, |det r| <= 2e-2): neutral ones kappa 0.493-0.5005 (|G| < 0.07), charged ones
  0.50-0.94.  (The 0.493-0.496 values sit at E_Z <= 0.03, phi pi: the gap-closing tip; eta absorption suspected, not
  checked.)  One non-converged (L 3, E_Z 0.01, |det r| 0.66) excluded.  61-83 % of the kappa peaks > 0.3 sit within
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

## SCHARF: "resonance, not a topological plateau" -- advisor figures, mixed SOC (2026-11-06, slides_nov.py)
* collapse_mixed (plots/slides/mixed_collapse.png; caches scharf_absOM_*, L 2 um, a 10, phi pi): NEGF kappa vs
  x = eps_box/Gamma with Gamma MEASURED (local Andreev Lorentzian, 1.15 / 3.18 ueV at tau 0.3387 / 0.5), no fit.
  E_Z <= 0.94 (separated pair): |kappa - 2x^2/(1+x^2)^2| median 0.013 / 0.006, 90 % 0.085 / 0.112; max kappa 0.501 /
  0.499; the visible outliers are the charged points (7 / 19 of 85, |G| > 0.1 kappa) and the slow Gamma drift.  The
  two taus put their kappa = 1/2 points at different fields but on one curve in x.
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
[2026-10-08/12, OLD bonds, a 20, alpha 14.3 / beta 7.3; disorder.py now inherits the ldos.PP sandbox -- see the
AUDIT note under STALE DATA in INTERFACE BONDS before rerunning anything.]
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
* Code: `computations/disorder.py` -- `measure(W, seed, E_Z, phi)`, `sweep(...)`,
  `load_or_run(name, ...)`; results in `plot_npz/scharf_disorder_<name>.npz`, arrays (W, seed, E_Z, phi).
* POINT A MOVED: disorder.py now has A = (E_Z 0.98 meV, phi = pi).  Every number in this
  Disorder section was measured at A = (0.505, pi).  At 0.98 the clean (old-bond) sweep has a narrow
  kappa peak 0.502 with |G|/kappa up to ~0.27 nearby (high-field, partly charged) -- check G there.

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
