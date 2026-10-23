# Context for Claude Code — thesis/fourterminal

Master's thesis code: NEGF transport through a four-terminal Josephson junction.
**TWO DIFFERENT SYSTEMS share this package.  Their numbers are NOT comparable — check which
one a statement belongs to before reusing it.**  Current focus: the SCHARF model.
See README.md for structure and usage.

| | Gresta / Domínguez | Scharf |
|---|---|---|
| paper | PRB 114, 125405 | PRB 99, 214503 |
| driven from | `Params` directly | `PhysParams` in `computations/ldos.py` |
| units | dimensionless, t = 1 | physical meV/nm, t = 2.507 meV (CODATA; was 2.488) |
| geometry | nx=80 × ny=12 | nx=100 × ny=5  (L=2000 nm × W=100 nm) |
| Zeeman | `Bz_s`=0.94, OUT-of-plane, in the RIBBONS | `Bxy_c` in-plane, in the CENTRE; `Bz_s`=0 |
| ribbons | TOPOLOGICAL, Chern = −1 | TRIVIAL, Chern = 0 |
| Majoranas | PROPAGATING chiral mode along the junction | LOCALISED pair at the ends y = ±L/2 |
| α | 1.2 (units of t·a) | 14.3 meV·nm = 0.2865 t·a |
| centre | DEPLETED, μ_c = −0.03 | POPULATED, μ_N = 0.7 meV |
| κ = 1/2 is | a PLATEAU, protected by a bulk gap | a RESONANCE, width = the MBS splitting δ |
| probe contact | transparent: tc ≈ 1.1, μ_n ≈ 1 | WEAK: old tc_barr 0.2 ≈ new 0.26, or a lattice barrier (barrier_n/V) |

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
* What it changes (computations/soc_bond_compare.py keeps the OLD variant, scaled=False):
  - OLD tc_barr = 0.2 behaves like NEW tc_barr ~ 0.26 = sqrt(0.2^2 + 0.16^2): the clean E_Z sweep
    reproduces the old resonance positions and heights.  So every Scharf weak-probe number below
    measured at "tc_barr = 0.2" (or 0.5) belongs to a stronger coupling than its label.
  - NEW tc_barr = 0.2 moves the resonances (0.43, 0.485, 0.53, 0.555, 0.625, ... meV); point A
    (E_Z 0.505) then sits on a flank, kappa 0.402.  PP's working point needs re-tuning.
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
* STALE DATA: computations/plot_npz/scharf_disorder_*.npz were computed with the OLD bonds.  Their
  kappa, G, andreev (weak probe) are pre-fix; kappa_s, G_s, andreev_s, det are unaffected.
  disorder.load_or_run will return them without noticing -- delete them to recompute.

## SCHARF model (`computations/ldos.py`; Scharf et al., PRB 99, 214503)
* MATCH TO THE PAPER (checked line by line against Eq. 1 / App. B Eq. B6, 2026-10-13): basis, kinetic
  term, the m(alpha^2+beta^2)/2hbar^2 shift, the [100] SOC hops, pairing sign, Zeeman (theta_code =
  pi - theta_Z), mu_N/mu_S, a = 20 nm, 5 x 100 N sites -- all EXACT.  Deliberate differences, kept:
  semi-infinite S ribbons (Scharf: W_S = 450 nm hard walls; phase boundaries barely differ, App. A2),
  Lorentzian eta (Scharf: Gaussian Gamma = 0.05 Delta on eigenstates), the transport probes (not in
  Scharf), and the phase on the TOP ribbon = Scharf's x < -W/2 side, i.e. phi -> -phi, compensated by
  the + pi on theta_z in PP (symmetry phi -> -phi with E_Z -> -E_Z).  Constants are CODATA since
  2026-10-13 (t 2.488 -> 2.507 meV, E_T = 832 ueV exactly); N_MID = 5 (100 nm bulk probe, Fig. 8g).
  All plot_npz results predate the constants change (0.8% in t).
* BOND SETTINGS THAT REPRODUCE SCHARF (keep these):
  - tc_top = tc_bot = 1: Scharf's S/N interface is a plain bulk bond; the mismatch is mu_N vs mu_S
    only.  With the whole-bond scaling, tc = 1 is bit-identical to the bulk bond, SOC included.
  - LDOS: tc_barr = 0 = Scharf's hard walls at y = +-L/2.  Only exact WITH the whole-bond scaling
    (the old bond left a 0.16 t spin-flip hop to the probes).
  - Transport (our addition): tc_barr < 1 is a reduced-transparency contact whose SOC scales with it
    (old tc_barr 0.2 ~ new 0.26).  For a contact made of the material itself use tc_barr = 1 with
    barrier_n / barrier_V.  Either way: barrier_V = 0 and tc_barr = 0 for anything meant as Scharf.
  - alpha_c / alpha_n / alpha_s = None (one global alpha, beta everywhere, as in Scharf).
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
  Maps cached in `computations/plot_npz/scharf_kappa_map_{100,110}.npz` (61×61, ~16 min each).
* soc_axis matters: '110' puts (α−β) on the propagation axis, '100' puts √(α²+β²). With β=7.3
  that halves the effective SOC at '110' — κ>0.05 on 9.6% of the map vs 28.6% at '100'.

## SCHARF: probe leads, mu_N vs mu_C, lattice constant (2026-10-14/19)
* NAMING CHANGED: in `PhysParams`, `mu_N` is now the PROBE LEADS (Params.mu_n) and `mu_C` the
  normal strip between the SCs (Params.mu_c; Scharf's "N").  `mu_C=None` means mu_C = mu_N (the
  old single-mu behaviour); `pp.mu_centre` returns the effective value and E_T / alpha k_F use it.
  EVERY "mu_N" in the sections above (0.7 meV working point, the 0.65-0.95 wedge, the subband
  edge) means the CENTRE, i.e. today's mu_C.  ldos.PP: mu_N = 3.0 (leads), mu_C = 0.7.
* WHY: the probes are W-wide hard-wall strips; at mu = 0.7 meV they sit at their first transverse
  subband (lattice 2t(1-cos(pi/(ny+1))): 0.672 / 0.812 / 0.896 meV at a = 20 / 10 / 5 nm,
  continuum ~0.99 at W = 100).  Open at a = 20 only by 0.03 meV -- a lattice artefact; CLOSED at
  a = 10 and 5 (kappa = 6e-11, Andreev 2e-12: a dead contact).  Opening the leads (mu_N = 3) at
  a = 5, L = 300 restores transport (kappa 0.73-0.78, but G ~ kappa: L < xi_MBS, transparent probe).
  So the "kappa = 0 below mu_N ~ 0.62" edge in the wedge section is the PROBE closing, not the strip.
* tc_barr / tc are DIMENSIONLESS (fraction of the lattice hop) -- never divide by t.  But a fixed
  tc_barr is not a fixed contact: one weak bond has T ~ 4 tc^2 (k_F a)^2, so it gets ~a^2 more
  opaque on refining the lattice.  An a-independent contact = physical barrier (V_b meV -> barrier_V
  = V_b/t, d nm -> barrier_n = round(d/a), tc_barr = 1) or the mu_N/mu_C density step.  NOT
  implemented as PhysParams fields yet.  tc = 1 (S/N) is a bulk bond and a-independent.
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
thermal, LDOS, run_scan with functools.partial; save_fig writes computations/plots/<name>.png).  Thin studies import ONLY it: `study_W.py`
(ABS count, gaps, k_x = 0, optional ldos map), `study_checks.py` (a, double peak,
long L), `study_delta.py` (+ 4 ABS at long L), `study_tc.py`, `study_T.py`
(finite T, tc at finite T), `study_transition.py` (LDOS transition, unfinished).  The old
scharf_abs*.py scripts and logs are in computations/archive/ (reference only; they no longer run
from there).  base_params() writes the pure-Rashba set out explicitly -- it no longer follows
ldos.PP.  Caches plot_npz/scharf_abs*_*.npz keep their old names (verified identical) and
do NOT encode parameters: delete them if parameters change.  The mixed-SOC and finite-strip
(H_device) code was dropped; it lives only in archive/scharf_abs.py.
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
* LARGER DELTA DOES WHAT WIDENING CANNOT (2026-10-20, `study_delta.py`, pure Rashba, a = 20,
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
  - 4 ABS (W = 300, Delta 0.75/1.0), `study_delta.py` cell 3: L = 4 um still CHARGED (peaks 0.44-0.66,
    |G|/kappa 0.17-0.36 -- where 2 ABS at Delta 0.25 was already neutral).  L = 8 um (Delta 0.75):
    GENUINE -- 14/17 peaks neutral, 0.46-0.49 (5 at >= 0.48), δ=0 double peaks, and NO neutral
    peak above 0.52 (single channel; contrast W = 800, 3 ABS, 8 um: neutral 0.56-0.68).
    So more ABS at fixed gap still lengthens the required L (2 ABS: 4 um, 4 ABS: 8 um at W = 300).
  - Caveat: Delta >= 0.75 approaches mu_S = 1 meV (strong-coupling SC); treat as a trend.
  Rule: ABS count ~ Delta/E_T, protecting gap ~ f(W); raise Delta for ABS, keep W small for the gap.
* k_x = 0 (normal incidence) vs ALL k (2026-10-20, `study_W.py` cell 3, pure Rashba, a = 20):
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
* THE PROBE COUPLING DOES NOT REPLACE LENGTH (2026-10-20, `study_tc.py`, tc_barr_transport
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
* LDOS THROUGH THE TRANSITION (2026-10-23, study_transition.py, L 2 um, phi pi, eta 0.05 Delta):
  W 300: inside Q = -1 the MIDDLE LDOS is gapped at E = 0 (a dark diamond following the k = 0 gap) while
  the END keeps zero-energy weight; real-space LDOS(E = 0) end/middle ~ 2.5 at the best point (0.323 meV).
  W 800: the middle shows the same checkerboard of levels as the end and the LDOS(E = 0) profile is flat
  (end/middle ~ 1.2): no end localisation.  NB eta = 12.5 ueV exceeds Delta_top at W 800 (10 ueV), so
  Scharf's broadening cannot resolve it there anyway; the 'trivial' and 'gap closing' picks of W 800
  both fall on E_Z = 0 (the first topological field is the second grid point).

## SCHARF: lattice-constant convergence and the probe contact (2026-10-23/24, `study_a.py`)
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
* PURE-RASHBA CASES SURVIVE ~10x HOTTER than the numbers above (different SOC, mu and field
  protocol -- the cause is NOT isolated) (2026-10-21, `study_T.py` cell 1, a = 20, tc_barr 0.2, best kappa over a
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
* THE COUPLING HAS A TEMPERATURE OPTIMUM (2026-10-21, `study_T.py` cell 2, W 100, L 2, phi = pi,
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
  (add the couplings to SCAN_CASES in study_T.py; the finished lines are cached in plot_npz/scharf_absTtc_*).
* MORE KNOBS AT FINITE T (2026-10-23, study_T.py cell 2; best NEUTRAL kappa, |G|/kappa < 0.05,
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
* The W lever for Scharf is UNTESTED and is NOT the Gresta ny lever: here W sets the Thouless
  energy, E_T = (pi/2)*2*sqrt(mu_N t)/n_across ~ 1/W, and the topological region sits at
  E_Z ~ E_T, so a WIDER junction needs a WEAKER field (W=100 nm -> 2.9 T, W=200 -> 1.4 T,
  W=300 -> 1.0 T).  Opposite direction to Gresta.  Whether that trades field for temperature
  favourably is open.

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
* THE RESULTING TENSION, both halves measured in this model:
      the quantity WITH a plateau (local 2e^2/h) CANNOT prove a Majorana -- disorder fakes it,
          with BETTER LDOS contrast (17-19) than the genuine MBS (3.2-6.4), at kappa ~ 1e-7;
      the quantity that CAN prove one (nonlocal kappa=1/2 with G=0) has NO plateau.
* RESOLVED (mu_N scanned): kappa oscillates in mu_N exactly as it does in E_Z, with quantised
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
* mu_N was the last untested parameter, and it CONFIRMS the plateau rather than shrinking it.
  At phi=pi, tc_barr=1.0, the local Andreev conductance holds 2e^2/h over a connected WEDGE:
  86 cells of a 29x31 map, mu_N = 0.65-0.95 meV x E_Z = 0.04-0.72 meV, G = 1.952-1.999
  (mean 1.9899, std 0.013).  The largest fully-inside RECTANGLE is
      mu_N = 0.65-0.90 meV  (0.25 meV)   x   E_Z = 0.08-0.40 meV  (0.32 meV),
  mean 1.9949 throughout.  PP.mu_N = 0.7 sits inside it.  mu_N is GATE-tunable, so this is a
  far stronger experimental claim than the 1D E_Z line: the plateau can be found by sweeping a
  gate at fixed field, and it does not require hitting a field value.
  Cached: `computations/plot_npz/scharf_mu_{map,line}.npz`, figure `computations/scharf_mu_map.png`.
* THE WEDGE CLOSES AS E_Z RISES: the mu_N width goes 0.30 meV (E_Z=0.12-0.32) -> 0.25 (0.36-0.40)
  -> 0.20 (0.44-0.48) -> 0.15 (0.52-0.56) -> 0.10 (0.60) -> 0 by E_Z=0.68.  So LOW field is
  better, which is the opposite of the usual "push the field up to get topological" instinct.
  I predicted the upper edge would GROW like sqrt(E_Z^2 - Delta^2) (a 1D-wire topological
  criterion); it does the reverse.  Unexplained -- do not quote a mechanism for it.
* THE LOWER EDGE IS THE FIRST TRANSVERSE SUBBAND, and it is E_Z-INDEPENDENT.  For a ny-site
  strip the subband bottoms are E_n = 2t(1-cos(n pi/(ny+1))); at ny=5, t=2.4877 meV that is
  0.667, 2.488, 4.975, ... meV.  Everything (kappa, G, Andreev) is EXACTLY ZERO below
  mu_N ~ 0.62 because the normal strip has no propagating mode at all.  The measured edge sits
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
