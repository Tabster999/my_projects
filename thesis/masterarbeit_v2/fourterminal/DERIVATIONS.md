# From the Hamiltonian to the observables: the formulas behind every number

This document covers every quantity the Scharf-model studies compute. For each one it gives the equation, a
derivation, the physical reasoning, a literature source where one exists, and the code that implements it.
Links point to the exact lines; in VS Code they open the file at that line, and the preview renders the
equations.

**What the source tags mean**
- **[source, verified]**: the formula is in the cited paper. It was checked against the paper's own LaTeX source
  from arXiv; equation numbers were checked against the printed PDF.
- **[follows from source]**: a few lines of algebra on verified equations, with every step written out. Where a
  paper states a limit or special case of the result, it is named.
- **[derived here]**: no paper states it, because it is either a convention of this code (axes, units, basis,
  storage) or analysis done for this project. The derivation is written out in full.
- **[model choice]**: a decision made in this code, not a physical law.

Everything is in the lattice units of `my_functions` unless stated otherwise (energies in units of the hopping
$t$, lengths in units of $a$). The conversion to meV and nm is §1.3.

---

## 0. The pipeline at a glance

```
PhysParams (meV, nm, T)          ldos.py          §1.3  units, E_Z <-> B, E_T, field angle
        |  .params() / .params_transport()
        v
Params (lattice units)  ->  4x4 BdG blocks        hamiltonians.py   §1    H = onsite + hoppings
        |
        +--> TRANSPORT GEOMETRY (finite centre, semi-infinite probes and SC ribbons)
        |      leads.py   §2.1  surface Green's functions (Sancho-Rubio), self-energies, Gamma
        |      rgf.py     §2.2  G of the centre by RGF + one exact Dyson step for the ribbons
        |      rgf.py     §3    channels T_ee, T_he, ... (Caroli formula)
        |      tools.py   §4    kappa, G, local Andreev, det r      transport.py §4.4  finite T
        |      study_*.py §5    two-Majorana model: x = eps/Gamma, Gamma_eff, p, plateau criterion
        |
        +--> CROSS-SECTION (infinitely long junction, Bloch k along it)
               tools.py           §6    H(k), gaps, Pfaffian invariant Q
               study_oscillation  §7    complex k (xi_M, k_osc), gap dip, closed box eps,
                                        overlap law, end modes (study_mixed [amplitude])
```

The two branches meet in §5. The closed-box splitting $\varepsilon$ from §7, set against the transport output of
§4, tests the two-Majorana picture without any free parameter.

---

## 1. The model

### 1.1 Continuum Hamiltonian
**[source, verified]** Scharf et al., PRB 99, 214503 (2019), arXiv:1904.08981, **Eq. (1)**. The basis is
$\Psi = (\psi_\uparrow, \psi_\downarrow, \psi_\downarrow^\dagger, -\psi_\uparrow^\dagger)$, the same order the
code uses:

$$
\begin{aligned}
H_\mathrm{BdG}={}&\Big[\tfrac{p_x^2+p_y^2}{2m}+\tfrac{\alpha}{\hbar}(s_yp_x-s_xp_y)+\tfrac{\beta}{\hbar}(s_xp_x-s_yp_y)
+\tfrac{m(\alpha^2+\beta^2)}{2\hbar^2}-\mu_S\Big]\tau_z\\
&+\big(V_0\tau_z-\mathbf E_Z\cdot\mathbf s\big)h(x)+\Delta(x)\big[\tau_x\cos\Phi(x)-\tau_y\sin\Phi(x)\big].
\end{aligned}
$$

The paper's $x$ is across the junction (width $W$) and its $y$ is along it (length $L$). $h(x)=1$ in the normal
(N) strip, and $\Delta(x)=\Delta$ only in the superconducting (S) regions. The paper writes the N potential as
$V_0=\mu_S-\mu_N$; the code instead gives each region its own $\mu$.

**Physics.**
- In this basis, kinetic energy, chemical potential and SOC are odd under particle–hole conjugation
  ($\propto\tau_z$), while the Zeeman term is even ($\propto\tau_0$).
- An in-plane field together with a phase difference near $\pi$ drives the quasi-1D N strip into class D
  topological superconductivity, with one Majorana at each end of the strip. Scharf, Sec. III: "As has been
  demonstrated in Ref. [Pientka et al., PRX 7, 021032 (2017)] … an in-plane Zeeman term … can induce
  topological superconductivity", in a region around $\phi=\pi$.

**The axis swap.** `my_functions` counts its $x$ along the interfaces. So package $x$ = paper $y$ and package
$y$ = paper $x$; the swap is the two lines [ldos.py:182-183](computations/ldos.py#L182-L183). **[derived here]**
Substituting $p_x^\mathrm{paper}=p_y$ and $p_y^\mathrm{paper}=p_x$ into Eq. (1) gives the SOC in package
coordinates ('100' setup):

$$
H_\mathrm{soc}=-(\alpha s_x+\beta s_y)\,k_x+(\beta s_x+\alpha s_y)\,k_y ,
$$

which is the form stated in [ldos.py:45-46](computations/ldos.py#L45-L46) and built in
[hamiltonians.py:79-80](my_functions/hamiltonians.py#L79-L80) and [:93-94](my_functions/hamiltonians.py#L93-L94).

### 1.2 Lattice discretisation
**[source, verified]** for the Rashba lattice: Mireles & Kirczenow, PRB 64, 024426 (2001), arXiv:cond-mat/0105049,
Sec. II.C.
- **Eq. (18)** is the tight-binding Rashba term: a nearest-neighbour hop $\mp t_\mathrm{so}(i\sigma_{y,x})$ with
  $t_\mathrm{so}=\alpha/2a$, next to the kinetic hop $|t|=\hbar^2/2m^*a^2$ and the onsite energy $4|t|$.
- **Eqs. (24)–(25)** show that its bands reduce to the continuum Rashba dispersion for small $ka$.

The code uses the same amplitudes: kinetic $t$, onsite $4t$, SOC hop $\alpha/2a$ (and $\beta/2a$ for Dresselhaus).
**[follows from source]** The continuum limit below repeats their check for the full block, which also covers the
Dresselhaus term and the BdG hole sector that they do not treat.

**Onsite block** [hamiltonians.py:25-47](my_functions/hamiltonians.py#L25-L47):

$$
h_\mathrm{on}=\big[4t-\mu+\tfrac{\alpha^2+\beta^2}{4}\big]\tau_z+E_Z(\cos\theta\,s_x-\sin\theta\,s_y)\,\tau_0
+\Delta\,(e^{i\phi}\tau_++\mathrm{h.c.}).
$$

Here $\alpha,\beta$ are in lattice units. The pairing sits on the elements (0,2) and (1,3): $\Delta e^{i\phi}$ in
the basis above. Scharf's $\Delta[\tau_x\cos\Phi-\tau_y\sin\Phi]$ is exactly
$\begin{pmatrix}0&\Delta e^{i\Phi}\\ \Delta e^{-i\Phi}&0\end{pmatrix}$ in $\tau$ space.

**Hopping to the $+x$ / $+y$ neighbour** [hamiltonians.py:54-64](my_functions/hamiltonians.py#L54-L64):

$$
V=-t\,\tau_z+M\oplus(-M),\qquad M=-\tfrac{i}{2}\,(c_x s_x+c_y s_y).
$$

The SOC block appears as $+M$ in the electron sector and $-M$ in the hole sector.

**Continuum limit.** With $H(\mathbf k)=h_\mathrm{on}+\sum_{d}\big(V_d e^{ik_d}+V_d^\dagger e^{-ik_d}\big)$:
- kinetic part: $4t-2t\cos k_x-2t\cos k_y\simeq t(k_x^2+k_y^2)$, i.e. $t\,a^2k^2=\hbar^2k^2/2m$ for
  $t=\hbar^2/2ma^2$;
- SOC part: $M e^{ik}+M^\dagger e^{-ik}=\sin k\,(c_xs_x+c_ys_y)\simeq k\,(\mathbf c\cdot\mathbf s)$, which is
  Eq. (1) after the axis swap;
- hole sector: $-s_y h^*(-\mathbf k)s_y=-h_\mathrm{soc}(\mathbf k)$ for a time-reversal-even SOC term, hence the
  $-M$;
- the constant $(\alpha^2+\beta^2)/4$ in lattice units equals $(\alpha^2+\beta^2)/(4a^2t)=m(\alpha^2+\beta^2)/2\hbar^2$
  in physical units. That is Scharf's SOC shift, so $\mu$ is measured from the bottom of the lower SOC band
  (CLAUDE.md, "constant SOC shift").

**Interfaces [model choice]** [hamiltonians.py:191-210](my_functions/hamiltonians.py#L191-L210). A bond of
strength $t_b$ between regions (probe bond $\tau$ = `tc_barr_transport`, S/N bond $t_c$) is:
- for $t_b\le1$: $t_b\times[-\tau_z+\mathrm{SOC}]$ (a barrier weakens the whole hop, spin rotation included);
- for $t_b>1$: $-t_b\tau_z+\mathrm{SOC}$.

At $t_b=1$ it is an ordinary bulk bond. Why this matters: CLAUDE.md, "INTERFACE BONDS SCALE AS A WHOLE".

**Assembly.**
- Central region: [junction.py:29-64](my_functions/junction.py#L29-L64). Probes: one column of $n_y$ sites per
  cell; SC ribbons: one row of $n_x$ sites per cell.
- Phase: $\phi$ sits on the top ribbon, $0$ on the bottom one
  ([junction.py:56-57](my_functions/junction.py#L56-L57)). That is a gauge-equivalent choice to Scharf's symmetric
  $\Phi(x)$.
- Zeeman field: only in the N strip (`Bxy_c`, [ldos.py:218](computations/ldos.py#L218)), as $h(x)$ in Eq. (1).
- Disorder (optional): $V_i\tau_z$ per site, [junction.py:81-102](my_functions/junction.py#L81-L102).

### 1.3 Units and derived scales
**Code:** [ldos.py:142-231](computations/ldos.py#L142-L231).

| quantity | formula | source | code |
|---|---|---|---|
| hopping | $t=\hbar^2/2ma^2$ (2.507 meV at $a=20$ nm, $m=0.038m_0$) | definition (§1.2) | [ldos.py:143](computations/ldos.py#L143) |
| lattice parameters | $\mu_t=\mu/t$, $\Delta_t=\Delta/t$, $\alpha_t=\alpha/(at)$ | follows from §1.2 | [ldos.py:208-221](computations/ldos.py#L208-L221) |
| Zeeman energy | $E_Z=g\mu_BB/2$ | **Scharf, Sec. VII** ("$\mathbf E_Z=g\mu_B\mathbf B/2$") | [ldos.py:148-154](computations/ldos.py#L148-L154) |
| Thouless energy | $E_T=\frac{\pi}{2}\frac{\hbar v_F}{W}$, $v_F$ of the N strip | **Scharf, Eq. (8)** and the text after it | [ldos.py:157-159](computations/ldos.py#L157-L159) |
| field angle | $\mathbf E_Z\perp\mathbf n_\mathrm{soc}$: $\theta_z=\arctan2(\alpha,\beta)$ ('100') | **[derived here]**, below | [ldos.py:161-165](computations/ldos.py#L161-L165) |
| mixed SOC | $(\alpha,\beta)=16\,(\cos0.15\pi,\ \sin0.15\pi)$ meV nm | [model choice] (the thesis case) | [tools.py:72-74](computations/tools.py#L72-L74) |
| temperature | $1\ \mu\mathrm{eV}/k_B=11.6$ mK | $k_B=86.17\ \mu$eV/K | [tools.py:355](computations/tools.py#L355) |

**Field angle [derived here].**
- For propagation along the junction ($k_x$), the SOC field is $\mathbf n_\mathrm{soc}\propto(\alpha,\beta)$ (§1.1).
- The code's Zeeman term points along $(\cos\theta,-\sin\theta)$.
- Perpendicular means $\alpha\cos\theta-\beta\sin\theta=0$, i.e. $\theta=\arctan2(\alpha,\beta)$.

The extra $+\pi$ in `case()` ([tools.py:68](computations/tools.py#L68)) reverses the field. That mirrors the maps
about $\phi=\pi$ and compensates the top/bottom orientation (CLAUDE.md, "MATCH TO THE PAPER").

**Thouless energy in lattice units [derived here].** $\hbar v_F=\hbar\sqrt{2\mu/m}=2a\sqrt{\mu t}$ and
$W=n_\mathrm{across}\,a$, so $E_T=\frac{\pi}{2}\cdot2\sqrt{\mu t}/n_\mathrm{across}$, which is the code line.

---

## 2. Green's functions of the transport geometry

### 2.1 Semi-infinite leads: surface Green's function, self-energy, coupling
**Equations.** A lead is a chain of identical cells: onsite block $H_0$, outward hop $V$. Its surface Green's
function $g_s=(z-\varepsilon_s)^{-1}$ with $z=E+i\eta$ satisfies the self-consistency (Dyson) equation

$$
\varepsilon_s=H_0+V\,(z-\varepsilon_s)^{-1}V^\dagger .
$$

**[source, verified]** Lewenkopf–Mucciolo (below), Sec. 4.1, **Eqs. (60)–(61)**: "Since adding another slice to a
semi-infinite lead should not alter its Green's function", $G^L_{1,1}=g_L$, which gives the self-consistency
condition $g_L=(E-h_1-t_x^2g_L)^{-1}$. That is the equation above, for a scalar hop $t_x$. The coupling to the
centre (hop $V_c$) gives the self-energy and the coupling matrix:

$$
\Sigma(z)=V_c^\dagger g_s(z)V_c,\qquad \Gamma=i(\Sigma-\Sigma^\dagger).
$$

**[source, verified]** Lewenkopf & Mucciolo, J. Comput. Electron. 12, 203 (2013), arXiv:1304.3934: **Eq. (16)**
gives $\Gamma_{L,R}=i[\Sigma^r-\Sigma^{r\dagger}]$, **Eq. (17)** gives $\Sigma^r_L=u_Lg^r_Lu_L^\dagger$, and
**Eq. (18)** defines the surface Green's function.

**Algorithm.** Sancho–Rubio decimation: M. P. López Sancho, J. M. López Sancho and J. Rubio, J. Phys. F 15, 851
(1985). That paper's own text was not checked; Lewenkopf–Mucciolo (Sec. 4.3) and Gresta et al. (ref. 61) cite it
for exactly this method. Each iteration eliminates every second cell and renormalises

$$
\varepsilon_s\leftarrow\varepsilon_s+\alpha g\beta,\qquad \varepsilon_b\leftarrow\varepsilon_b+\alpha g\beta+\beta g\alpha,
\qquad \alpha\leftarrow\alpha g\alpha,\quad \beta\leftarrow\beta g\beta,\qquad g=(z-\varepsilon_b)^{-1},
$$

starting from $\alpha=V$, $\beta=V^\dagger$. The effective chain length doubles at every step.

**[source, verified]** Lewenkopf–Mucciolo print this recursion in Sec. 4.3, **Eqs. (86)–(93)**.
- **Their Eq. (91) has a typo.** It reads $\varepsilon^s_k=\varepsilon_{k-1}+\alpha_{k-1}g\,\beta_{k-1}$. Inserting
  their Eq. (87) for site $2^{k-1}$ into their Eq. (86) gives
  $(z-\varepsilon^s_{k-1}-\alpha g\beta)G_{0,0}=1+\alpha g\alpha\,G_{2^k,0}$, so the surface term builds on itself:
  $\varepsilon^s_k=\varepsilon^s_{k-1}+\alpha_{k-1}g\,\beta_{k-1}$, as written above. Their Eq. (92) for the bulk
  term is correct.
- The code uses the correct form ([leads.py:71](my_functions/leads.py#L71)). The residual check below would also
  catch a wrong surface term, since it tests the converged $\varepsilon_s$ against the self-consistency equation.

**Physics.**
- $\Sigma$ is what the infinite lead does to the centre. Its Hermitian part shifts levels; its anti-Hermitian part
  $\Gamma$ is the escape rate into the lead, the broadening of a level coupled to it.
- The small $\eta$ acts as a uniform absorption. It must stay far below every physical width; CLAUDE.md
  ("Numerics") has the measured consequences.

**Code**
- decimation: [leads.py:48-80](my_functions/leads.py#L48-L80);
- residual check of the Dyson equation above: [leads.py:83-87](my_functions/leads.py#L83-L87);
- Newton refinement: [leads.py:90-120](my_functions/leads.py#L90-L120);
- eigenmode fallback: [leads.py:123-135](my_functions/leads.py#L123-L135);
- failed solves become NaN and are counted in `N_UNRELIABLE`: [leads.py:174-179](my_functions/leads.py#L174-L179);
- $\Sigma$: [leads.py:239-242](my_functions/leads.py#L239-L242);
- $\Gamma$: [rgf.py:293-294](my_functions/rgf.py#L293-L294).

### 2.2 Green's function of the centre: RGF plus one exact Dyson step
**Equation.** The quantity needed is

$$
G^r=\big(z-H_C-\Sigma_L-\Sigma_R-\Sigma_T-\Sigma_B\big)^{-1},
$$

but only on the probe columns ($G_{LL},G_{LR},G_{RL},G_{RR}$).

**Step 1: RGF along the junction.** $A_0=z-H_C-\Sigma_L-\Sigma_R$ is block-tridiagonal in the columns, so
$g_0=A_0^{-1}$ follows from left and right sweeps:

$$
g^L_i=\big(A_{ii}-V^\dagger g^L_{i-1}V\big)^{-1},\qquad g^R_i=\big(A_{ii}-Vg^R_{i+1}V^\dagger\big)^{-1}.
$$

The off-diagonal blocks follow from $G_{ij}=g^R_iV^\dagger G_{i-1,j}$ for $i>j$ and $G_{ij}=g^L_iVG_{i+1,j}$ for
$i<j$.
- **[source, verified]** Lewenkopf–Mucciolo, Sec. 3:
  - the recursions are built from the Dyson equations $G=G^{(0)}+G^{(0)}VG$ and $G=G^{(0)}+GVG^{(0)}$, their
    **Eqs. (19)–(20)**;
  - the left and right recursions above are their **Eqs. (32)** and **(35)**;
  - the neighbouring off-diagonal blocks are their **Eqs. (38)–(39)**. The code applies them repeatedly.
- **Code:** [rgf.py:115-169](my_functions/rgf.py#L115-L169); the recursions are
  [:138-144](my_functions/rgf.py#L138-L144).

**Step 2: the two SC ribbons in one Dyson step.** **[follows from source]** This is Dyson's equation, LM Eq. (19),
with $G^{(0)}=g_0$. The perturbation is $\Sigma=\Sigma_T+\Sigma_B=W^\dagger g_sW$, which lives only on the boundary
rows $P$.
1. On $P$: $G_{PQ}=g_{0,PQ}+g_{0,PP}\Sigma G_{PQ}$, so $G_{PQ}=(1-g_{0,PP}\Sigma)^{-1}g_{0,PQ}$.
2. On $Q$: $G_{QQ}=g_{0,QQ}+g_{0,QP}\Sigma G_{PQ}$.
3. The push-through identity
   $W^\dagger g_sW(1-g_{0,PP}W^\dagger g_sW)^{-1}=W^\dagger(g_s^{-1}-Wg_{0,PP}W^\dagger)^{-1}W$ then gives

$$
G_{QQ}=g_{0,QQ}+\big(g_{0,QP}W^\dagger\big)\big[g_s^{-1}-Wg_{0,PP}W^\dagger\big]^{-1}\big(Wg_{0,PQ}\big),
\quad Q=\text{probe columns},\ P=\text{ribbon rows}.
$$

- It uses $g_s^{-1}=z-\varepsilon_s$, which stays finite even when the ribbon has a Majorana pole at $E=0$. That is
  why it is stable (docstring, [rgf.py:25-34](my_functions/rgf.py#L25-L34)).
- **Code:** [rgf.py:342-350](my_functions/rgf.py#L342-L350); its pieces are built in
  [:317-319](my_functions/rgf.py#L317-L319).
- **Checked:** the RGF channels agree with a dense inversion of the whole centre to $10^{-8}$, at
  $\phi=0,\,1.3,\,\pi$ and for both directions ([checks.py:55-67](my_functions/checks.py#L55-L67)). That covers
  this step and the phase gauge below.

**The phase costs nothing [derived here].** $\phi$ enters only the top ribbon's pairing. A gauge transformation
$U=\mathrm{diag}(1,1,e^{i\delta\phi},e^{i\delta\phi})$ per site maps the ribbon at $\phi_\mathrm{ref}$ to the
ribbon at $\phi$, and $U$ commutes with the normal-state coupling $W$, so

$$
g_T^{-1}(\phi)=U^\dagger g_T^{-1}(\phi_\mathrm{ref})\,U .
$$

The ribbon decimation therefore runs once; see [rgf.py:337-340](my_functions/rgf.py#L337-L340).

---

## 3. Transmission channels (Caroli formula in Nambu space)

**[source, verified]** Gresta et al., PRB 114, 125405 (2026), arXiv:2602.12947, **Eq. (7)**:

$$
\mathcal T^{\alpha\beta}_{RL}(\varepsilon)=\mathrm{Tr}\big[\Gamma_R\,\mathcal G^{r;\alpha\beta}_{RL}\,\Gamma_L\,
\mathcal G^{a;\beta\alpha}_{LR}\big],
$$

"associated with a quasiparticle scattering from type $\beta\in\{e,h\}$ from $L$ … to a type $\alpha$ in $R$". The
trace runs over spin and position. This is the Caroli formula (Caroli et al., J. Phys. C 4, 916 (1971)); for the
normal-state version see also Lewenkopf–Mucciolo, **Eq. (15)**.

**Physics.** $\Gamma$ is block-diagonal in electron/hole because the probes are normal metals. Projecting both
$\Gamma$'s and $G$ onto the e or h sectors splits the total transmission into processes:

| key in the code | process | physical meaning |
|---|---|---|
| `ee`, `hh` | $e\to e$, $h\to h$ across | elastic cotunnelling (EC) |
| `he_cross`, `eh_cross` | $e\to h$, $h\to e$ across | crossed Andreev reflection (CAR): a Cooper pair is split between the two probes |
| `he_local`, `eh_local` | $e\to h$ in the same probe | local Andreev reflection (LAR) |

**Particle–hole symmetry.** **[source, verified]** Klees et al., PRB 110, 064517 (2024), arXiv:2306.17845,
Sec. III: "$\mathcal T^{\tau_1\tau_2}_{\ell_1\ell_2}(\varepsilon)=\mathcal T^{\bar\tau_1\bar\tau_2}_{\ell_1\ell_2}(-\varepsilon)$".
At $E=0$ this forces `ee` = `hh` and `he_cross` = `eh_cross` exactly. The code uses this as a regression check
([transport.py:167-191](my_functions/transport.py#L167-L191)); it is not a breakdown detector.

**Code**
- RGF solver: [rgf.py:352-386](my_functions/rgf.py#L352-L386); the six formulas are
  [:373-380](my_functions/rgf.py#L373-L380);
- dense reference implementation, for comparison: [junction.py:215-234](my_functions/junction.py#L215-L234).

**Direction.** The studies call `channels_at_phi(phi, 'left')`, i.e. transmission from $R$ into $L$.
- **[derived here]** For the heat signal the direction does not matter at $E=0$ as long as only the two probes
  carry open channels (the trivial ribbons are gapped). Unitarity of the $L$/$R$ block of $S$ gives
  $\mathrm{Tr}\,tt^\dagger=\mathrm{Tr}\,t't'^\dagger$, and particle–hole symmetry makes each trace equal to
  $2(\mathcal T^{ee}+\mathcal T^{he})$.

---

## 4. Observables at the probes

### 4.1 Nonlocal charge conductance $G$
**[source, verified]** Gresta et al., **Eqs. (3) and (5)**:

$$
G=G_0\!\int\! d\varepsilon\Big(-\frac{\partial f}{\partial\varepsilon}\Big)\mathcal T^E,\qquad
\mathcal T^E=\mathcal T^{ee}-\mathcal T^{he},\qquad G_0=e^2/h .
$$

At $T\to0$ this gives $G/G_0=\mathcal T^{ee}-\mathcal T^{he}$ at $E=0$.

**Sign [follows from source].** The charge current of Klees et al., App. C (the final $I_R$, $J_R$ formula, after their
particle–hole simplification) is

$$
I_R=\frac{e}{h}\!\int\!d\varepsilon\,\big[\mathcal T^{ee}_{RL}(f^e_R-f^e_L)+\mathcal T^{eh}_{RL}(f^e_R-f^h_L)+\mathcal T^{eh}_{RR}(f^e_R-f^h_R)\big].
$$

Differentiating with respect to $V_L$ at $T=0$ gives $\partial I_R/\partial V_L=-G_0(\mathcal T^{ee}-\mathcal T^{he})(0)$.
So the code's `G` $=\mathcal T^{ee}-\mathcal T^{he}$ is Gresta's $G$, which is minus the nonlocal $dI/dV$ in units
of $e^2/h$.

**Physics.** EC carries charge $+e$ across, CAR carries $-e$. A Majorana pair transfers an electron as an
electron or as a hole with **equal** probability. Nilsson et al.: "with equal probabilities $p$", see §5.1. So
$G=0$ exactly. Charge neutrality is the Majorana fingerprint that the code tests as $|G|<0.1\,\kappa$.

**Code:** `G = ee − he_cross`, [tools.py:290](computations/tools.py#L290). The finite-bias version is
`current()` [transport.py:52-66](my_functions/transport.py#L52-L66), with the $\tfrac12$ that removes the Nambu
double counting.

### 4.2 Thermal conductance $\kappa$
**[source, verified]** Gresta et al., **Eqs. (4) and (6)**:

$$
\mathcal T^Q=\mathcal T^{ee}+\mathcal T^{he},\qquad \kappa_0=\frac{\pi^2k_B^2T}{3h},
$$

with the stated $T\to0$ limit $\kappa/\kappa_0=\mathcal T^Q$.

**[source, verified]** Klees et al., Sec. IV: the heat transport coefficient uses
$\mathcal T_{22}=\mathcal T^{ee}_{RL}+\mathcal T^{eh}_{RL}$. In their words, "a temperature gradient does not
generate LAR contributions to the thermal current $J_R$".

**Derivation [follows from source]** (from the $J_R$ line of Klees et al., App. C):

1. Take a thermal bias only: $\mu_L=\mu_R=0$, so for each probe $f^e=f^h=f_T$.
2. The LAR term $\propto f^e_R-f^h_R$ then vanishes identically, and
   $$J_R=\frac1h\!\int\!d\varepsilon\,\varepsilon\,\big[\mathcal T^{ee}_{RL}+\mathcal T^{eh}_{RL}\big](f_R-f_L).$$
3. Use $\partial f/\partial T=(\varepsilon/T)(-\partial f/\partial\varepsilon)$ and $x=\varepsilon/k_BT$:
   $$\kappa=\frac{k_B^2T}{h}\int\!d\varepsilon\,x^2\Big(-\frac{\partial f}{\partial\varepsilon}\Big)\mathcal T^Q
   \quad\Rightarrow\quad
   \frac{\kappa}{\kappa_0}=\int\!dx\;\frac{3}{\pi^2}\,\frac{x^2}{4\cosh^2(x/2)}\;\mathcal T^Q(xk_BT).$$
4. The kernel integrates to one, because $\int x^2/(4\cosh^2(x/2))\,dx=\pi^2/3$. So $T\to0$ gives
   $\kappa/\kappa_0=\mathcal T^Q(0)$.
5. $\mathcal T^{eh}_{RL}$ (Klees) and $\mathcal T^{he}_{RL}$ (Gresta) differ only by $\varepsilon\to-\varepsilon$
   (particle–hole symmetry). They are equal at $E=0$ and give the same integral with the even kernel.

> **Note on Gresta's Eq. (4) as printed.** It reads $\kappa=\kappa_0\int(\varepsilon/k_BT)^2(-\partial f/\partial
> \varepsilon)\mathcal T^Q$. Since the kernel integrates to $\pi^2/3$, that would give $\kappa/\kappa_0\to\frac{\pi^2}{3}
> \mathcal T^Q$ at $T\to0$, contradicting their own stated limit. The prefactor should be $k_B^2T/h=3\kappa_0/\pi^2$.
> The code uses the normalised kernel above, consistent with the stated limit.

This $\kappa$ is taken at zero voltage. The open-circuit value adds a thermoelectric correction
($K=3K_0\det\mathcal L/(\pi^2\mathcal L_{11})$ in Klees et al.), which Gresta neglects and so does the code.

**Physics.**
- A spin-degenerate normal channel with $T=1$ gives $\mathcal T^{ee}=2$, so $\kappa=2\kappa_0$.
- One Majorana carries half a fermionic degree of freedom. Gresta et al., Sec. I: "A single Majorana bound state
  carries only half of the degrees of freedom of a conventional fermionic mode … the thermal conductance reaches
  the universal quantized value, $\kappa=0.5\kappa_0$". This is the chiral (Gresta) case.
- In the localized-pair (Scharf) case, ½ is reached only on resonance (§5.1).

**Code:** `kappa = ee + he_cross`, [tools.py:290](computations/tools.py#L290);
`linear_response` [compute.py:35](my_functions/compute.py#L35); finite-$T$ kernels in §4.4.

### 4.3 Local Andreev conductance
**[follows from source]** from the LAR term of `current()`, [transport.py:63](my_functions/transport.py#L63), which
is the $\mathcal T^{eh}_{RR}(f^e_R-f^h_R)$ term of Klees et al., App. C (§4.1):
- The current is $I^\mathrm{LAR}=\tfrac12\int(R_{eh}+R_{he})(f_e-f_h)$, and $\partial(f_e-f_h)/\partial V=2(-\partial f/\partial E)$.
- Hence $G_\mathrm{LAR}=G_0\int(R_{eh}+R_{he})(-\partial f/\partial E)\to G_0\,(R_{eh}+R_{he})(0)=2G_0R_{he}(0)$.

**Physics.** An isolated Majorana gives perfect Andreev reflection, $R_{he}(0)=1$, hence $2e^2/h$.
- **[source, verified]** Law, Lee & Ng, PRL 103, 237001 (2009), arXiv:0907.1909: "the presence of a vortex in the
  superconductor changes the conductance to $2e^2/h$" (resonant Andreev reflection via a Majorana).
- **[source, verified]** Flensberg, PRB 82, 180516 (2010), arXiv:1009.3533, **Eq. (13)**: for a single isolated
  Majorana, $dI/dV=\frac{2e^2}{h}\frac{4\Gamma_F^2}{(eV)^2+4\Gamma_F^2}$. That is $2G_0R_{he}(eV)$ with the
  Lorentzian of §5.2, so the formula above holds at every energy, not only at $E=0$.
- This quantity needs only one Majorana, so it is not sensitive to the splitting. Its plateau is real but not
  specific to Majoranas (CLAUDE.md, "THE PLATEAU THAT DOES EXIST").

**Code:** `andreev = eh_local + he_local`, [tools.py:290](computations/tools.py#L290); finite $T$ in
[tools.py:376-385](computations/tools.py#L376-L385).

### 4.4 Finite temperature
**Equations:** the kernels of §4.1 and §4.2,

$$
W_\mathrm{el}(x)=\frac{1}{4\cosh^2(x/2)},\qquad W_\mathrm{th}(x)=\frac{3}{\pi^2}x^2W_\mathrm{el}(x).
$$

They are evaluated with the trapezoid rule in $x$, and $E<0$ is folded onto $E>0$ by particle–hole symmetry. That
is why `thermal()` sums `ee+he+hh+eh` over $E\ge0$, with weight ½ at $T=0$.
- **Code:** [transport.py:118-164](my_functions/transport.py#L118-L164) and
  [tools.py:370-391](computations/tools.py#L370-L391).
- Grid rule: the energy grid must reach about $16\,k_BT$.

### 4.5 $\det r$: the topological number seen by a probe
**Equations.**
- **[source, verified]** Lewenkopf–Mucciolo, **Eqs. (5) and (8)** (Fisher–Lee on a lattice):
  $S_{ab}=-\delta_{ab}+i\sqrt{\gamma_a\gamma_b}\,\langle\chi_a|G^r|\chi_b\rangle$, with $\Gamma=\sum_a|\chi_a\rangle\gamma_a\langle\chi_a|$.
- In the eigenbasis of $\Gamma_L$, restricted to its open channels, the left-probe reflection matrix is therefore

$$
r=-1+i\,\Gamma_L^{1/2}\,G_{LL}\,\Gamma_L^{1/2}.
$$

- **[source, verified]** Fulga, Hassler, Akhmerov & Beenakker, PRB 83, 155429 (2011), arXiv:1101.1749, **Table I**:
  for class D the topological quantum number is $\mathcal Q=\mathrm{sign\,Det}\,r$, with $r$ real in the Majorana
  basis. Sec. III: "The determinant is unchanged by the change of basis". That is why the code may take $\det r$ in
  the $\Gamma$ eigenbasis, and why the result is real up to rounding.

**Physics.**
- $\det r=-1$: the probe sees one isolated Majorana (perfect Andreev reflection, §4.3).
- $\det r=+1$: it sees none, or a hybridised pair.
- **[source, verified]** $\det r=0$ means one channel is fully transmitted. Akhmerov, Dahlhaus, Hassler, Wimmer &
  Beenakker, PRL 106, 057001 (2011), arXiv:1009.5542:
  - **Eq. (3):** $\mathrm{Det}\,r=\prod_n\tanh\lambda_n$, with transmission eigenvalues $T_n=1/\cosh^2\lambda_n$;
  - the text after it: "at the topological phase transition one of the $\lambda_n$'s vanishes … so the
    corresponding transmission eigenvalue $T_n=1$";
  - their thermal conductance is $G_\mathrm{th}=G_0\sum_nT_n$ with $G_0=\pi^2k_B^2\tau_0/6h=\kappa_0/2$.

  So $\kappa/\kappa_0=\tfrac12\sum_nT_n\ge\tfrac12$ at $\det r=0$. The sum runs over the e and h channels and counts
  each process twice.
- In a finite junction $\det r$ follows the Majoranas' overlap, not only the bulk phase (§5.1).

**Code:** [tools.py:266-277](computations/tools.py#L266-L277).

---

## 5. The two-Majorana model and what is extracted from it

### 5.1 Scattering through two coupled Majoranas
**[source, verified]** Nilsson, Akhmerov & Beenakker, PRL 101, 120403 (2008), arXiv:0806.4889:
- **Eq. (1):** $H_M=iE_M\gamma_1\gamma_2$, i.e. a splitting $\pm E_M$.
- **Eq. (2)** (Mahaux–Weidenmüller): $S(E)=1+2\pi iW^\dagger(H_M-E-i\pi WW^\dagger)^{-1}W$.
- **Eq. (3):** lead $i$ couples only to $\gamma_i$, with equal amplitude to electrons and holes.
- **Eqs. (4)–(6):**

$$
\begin{gathered}
S=\begin{pmatrix}1+A&A\\A&1+A\end{pmatrix},\qquad
A=\frac{1}{Z}\begin{pmatrix}i\Gamma_1(E+i\Gamma_2)&-E_M\sqrt{\Gamma_1\Gamma_2}\\E_M\sqrt{\Gamma_1\Gamma_2}&i\Gamma_2(E+i\Gamma_1)\end{pmatrix},\\
Z=E_M^2-(E+i\Gamma_1)(E+i\Gamma_2),\qquad \Gamma_i=2\pi w_i^2 .
\end{gathered}
$$

The general form of Eq. (2) and the particle–hole constraint $S(0)$ real orthogonal for $\mathcal C^2=+1$ are also
in Beenakker, RMP 87, 1037 (2015), arXiv:1407.2131, section on the scattering matrix (Cayley transform).

**At $E=0$ [follows from source]**, writing $x^2=E_M^2/(\Gamma_1\Gamma_2)$, so $Z=E_M^2+\Gamma_1\Gamma_2$:

$$
\mathcal T^{ee}=\mathcal T^{he}=|A_{21}|^2=\frac{x^2}{(1+x^2)^2}
\;\Rightarrow\;
\boxed{\kappa/\kappa_0=\frac{2x^2}{(1+x^2)^2},\quad G=0}
$$

$$
A_{11}=-\frac{1}{1+x^2}\;\Rightarrow\;
\det r=(1+A_{11})^2-A_{11}^2=1+2A_{11}=\boxed{\frac{x^2-1}{x^2+1}}
$$

$$
\Rightarrow\;\boxed{\kappa=\tfrac12\big(1-\det r^2\big)}\qquad\text{and}\qquad x=\sqrt{\frac{1+\det r}{1-\det r}} .
$$

Unitarity check: $R_{ee}+R_{he}+\mathcal T^{ee}+\mathcal T^{he}=[x^4+1+2x^2]/(1+x^2)^2=1$.

What the papers state themselves:
- **Nilsson et al., the text after their Eq. (8):** for $E,\Gamma_i\ll E_M$ ($x\gg1$), an electron goes to the other
  lead as an electron or a hole "with equal probabilities $p=\Gamma_1\Gamma_2/E_M^2$". That is the $x\gg1$ limit
  $\mathcal T^{ee}=\mathcal T^{he}\to1/x^2$ of the closed form above. The closed form itself is not in the paper.
- **Akhmerov et al., Eq. (3)** (§4.5): $\kappa=\tfrac12(1-\det r^2)$ also follows from it, without the two-Majorana
  algebra. A lead coupled to only one Majorana has every channel but one fully reflected ($|\tanh\lambda_n|=1$).
  The remaining channel then has $\tanh\lambda=\pm\det r$, so $T=1-\det r^2$ and $\kappa/\kappa_0=\tfrac12T$.

**Physics.**
- $\kappa$ reaches ½ **only** at $x=1$ ($\varepsilon=\Gamma$) and is smaller on both sides.
- $\varepsilon\to0$ gives $\kappa\to0$: the two Majoranas decouple and no heat crosses.
- $\varepsilon\gg\Gamma$ gives $\kappa\to0$: the pair is an ordinary off-resonant level.
- So $\kappa=\tfrac12$ is a **resonance in $\varepsilon/\Gamma$**, not a protected plateau (CLAUDE.md, "resonance,
  not a topological plateau").

**Finite energy [follows from source]** (used by the fingerprint finite-energy cell). For $\Gamma_1=\Gamma_2=\Gamma$:
- $\kappa(E)=2E_M^2\Gamma^2/|Z|^2$ with $|Z|^2=(E_M^2-E^2+\Gamma^2)^2+4E^2\Gamma^2$.
- Minimising over $u=E^2$ gives $E^*=\sqrt{E_M^2-\Gamma^2}$ and $|Z|^2_\mathrm{min}=4E_M^2\Gamma^2$.
- So $\max_E\kappa(E)=\tfrac12$ for every $E_M\ge\Gamma$.
- Code: `resonance_point`, [study_oscillation.py:476](computations/study_oscillation.py#L476).

**Code that uses these relations**
- $x$ from $\det r$: [study_tuning.py:458-467](computations/study_tuning.py#L458-L467) (`x_of`) and
  [study_mixed.py:151](computations/study_mixed.py#L151);
- $\kappa(x)$: [study_oscillation.py:453](computations/study_oscillation.py#L453) (`kappa_model`) and the collapse
  figure [slides_nov.py:417](computations/slides_nov.py#L417).

### 5.2 Measuring $\Gamma$ directly: the Andreev Lorentzian
**[follows from source]** from Nilsson Eqs. (4)–(6) with $E_M=0$:

$$
A_{11}=\frac{-i\Gamma_1}{E+i\Gamma_1}\;\Rightarrow\;R_{he}(E)=|A_{11}|^2=\frac{\Gamma_1^2}{E^2+\Gamma_1^2}.
$$

So $\Gamma$ is the **half width at half maximum** of the local Andreev signal at an isolated Majorana (at an
$\varepsilon$ zero). It is the same $\Gamma$ that enters $x$ in §5.1, so the collapse plot uses no fitted
parameter.

**[source, verified]** Flensberg, **Eq. (13)**, gives the same line shape for the conductance into an isolated
Majorana: $dI/dV=\frac{2e^2}{h}\frac{4\Gamma_F^2}{(eV)^2+4\Gamma_F^2}$, with height $2e^2/h$ and half width
$2\Gamma_F$. His coupling $\Gamma_F$ (**Eq. (9)**) is normalised differently: $\Gamma=2\Gamma_F$.
- **Code:** `gamma_point` [study_mixed.py:88](computations/study_mixed.py#L88) and `gamma_job`
  [study_tuning.py:489](computations/study_tuning.py#L489) bisect, in $\ln E$, for where
  (`eh_local`+`he_local`)/2 falls to half its $E=0$ value.

### 5.3 Effective coupling, exponent, and the plateau criterion (definitions)
- **$\Gamma_\mathrm{eff}=\varepsilon_\mathrm{box}/x(\det r)$**, on two-Majorana points only (neutral, and relation
  §5.1 within 0.01): [study_mixed.py:146-152](computations/study_mixed.py#L146-L152).
  - Physics: if transport sees the box splitting, $\Gamma_\mathrm{eff}$ must stay smooth while $\varepsilon$ varies
    by decades.
- **$\Gamma\propto\tau^p$:** $p=\ln(x_a/x_b)/\ln(\tau_b/\tau_a)$ at the same points of two maps,
  [study_tuning.py:470-476](computations/study_tuning.py#L470-L476).
  - Golden-rule expectation **[follows from source]**:
    - Flensberg **Eq. (9)** gives $\Gamma=2\pi\sum_{k\sigma}|V_{k\sigma}|^2\delta(\omega-\varepsilon_{k\sigma})$, with
      $V_{k\sigma}=\int dx\,f_\sigma(x)\,t_k(x)$ (Majorana wavefunction times tunnel amplitude, his Eq. (3) and the
      text after it).
    - The probe bond makes $t_k\propto\tau$, so at fixed wavefunctions $\Gamma\propto\tau^2$, i.e. $p=2$.
    - Nilsson's $\Gamma_i=2\pi w_i^2$ is the same statement.
  - Measured: $p\approx2.6$. Why it differs from 2 has **not** been established.
- **Plateau point [model choice]:**
  - $|\kappa-\tfrac12|<0.03$ and $|G|<0.1\kappa$: [study_tuning.py:394-396](computations/study_tuning.py#L394-L396);
  - the stricter version, which also requires $Q=-1$ and $|\kappa-\tfrac12(1-\det r^2)|<0.01$:
    [study_mixed.py:266-269](computations/study_mixed.py#L266-L269).
  - It marks an $\varepsilon\approx\Gamma$ region, never a protected plateau.

---

## 6. The cross-section: the infinitely long junction

### 6.1 Bloch Hamiltonian and gaps
**[derived here]** The junction is translation-invariant along $x$ (Bloch $k$, units $1/a$). Across it there are
$n_S$ SC sites, the N strip and $n_S$ SC sites, with hard walls ($W_S=900$ nm per side; Scharf uses finite $W_S$
too). The Bloch Hamiltonian is

$$
H(k)=\bigoplus_j\big(h_j+V_{x,j}e^{ik}+V_{x,j}^\dagger e^{-ik}\big)+\text{(across hops)}.
$$

It is built from the same blocks as the transport code ([tools.py:97-120](computations/tools.py#L97-L120)).
- Topological gap (Scharf's $\Delta_\mathrm{top}$): $\min_k|E(k)|$.
- The gap dip is about $1/\xi_M$ wide, so the grid minimum is refined by a 1D minimisation:
  [tools.py:187-220](computations/tools.py#L187-L220).

### 6.2 The invariant $Q$
**[source, verified]** Kitaev, Phys.-Usp. 44, 131 (2001), arXiv:cond-mat/0010440, **Eq. (26)**:

$$
\mathcal M(H)=\mathrm{sgn}\,\mathrm{Pf}\,\tilde B(0)\;\mathrm{sgn}\,\mathrm{Pf}\,\tilde B(\pi),
$$

where $\tilde B(q)$ is the real antisymmetric Majorana-basis Hamiltonian at the two particle–hole-invariant
momenta. $\mathcal M=-1$ means unpaired Majoranas at the ends. Scharf, Sec. III, footnote: the quasi-1D junction
is in class D and its invariant is the ground-state parity at $k=0$, citing Kitaev.

**[derived here]** The Majorana basis for $\Psi=(c_\uparrow,c_\downarrow,c_\downarrow^\dagger,-c_\uparrow^\dagger)$
is the matrix `_W1` ([tools.py:152-155](computations/tools.py#L152-L155)):
- rows give $\gamma_1=(c_\uparrow+c_\uparrow^\dagger)/\sqrt2$, $\gamma_2=i(c_\uparrow-c_\uparrow^\dagger)/\sqrt2$, and the
  same for spin down;
- $M(k)=-iW H(k)W^\dagger$ is real antisymmetric at $k=0,\pi$;
- a convention-dependent overall sign of $\mathrm{Pf}$ is the same at both momenta, so it cancels in $Q$.

**Code**
- $Q$: [tools.py:177-184](computations/tools.py#L177-L184);
- Pfaffian sign by Parlett–Reid elimination with pivoting: [tools.py:158-174](computations/tools.py#L158-L174).
  **[source, verified]** Wimmer, ACM TOMS 38, 30 (2012), arXiv:1102.3440, section "$LTL^T$ decomposition of dense
  matrices using the Parlett-Reid algorithm".

**Physics.** $Q$ belongs to the infinite junction: it does not know $L$ or the probes. $\det r$ (§4.5) is the same
invariant measured through a probe, and it tends to $Q$ only for $L\gg\xi_M$.

---

## 7. Lengths and the splitting $\varepsilon$

### 7.1 Complex momenta: decay length $\xi_M$ and oscillation $k_\mathrm{osc}$
**[source, verified]** Groth, Wimmer, Akhmerov & Waintal (Kwant), New J. Phys. 16, 063065 (2014), arXiv:1309.2926,
Sec. 2.2, **Eqs. (5)–(6)**:
- the modes of a translation-invariant lead are $\phi_n(j)=\lambda_n^j\chi_n$ with
  $(H_L+V_L\lambda_n^{-1}+V_L^\dagger\lambda_n)\chi_n=E\chi_n$;
- "the modes with $|\lambda_n|<1$ are evanescent".

Steps 1–2 below are this equation at $E=0$, with $H_1$ the hop to the next slice. The linearisation (step 3) and
the definitions of $\xi_M$ and $k_\mathrm{osc}$ (step 4) are **[derived here]**.
1. With nearest-neighbour hopping along the junction, $H(k)=H_0+H_1e^{ik}+H_1^\dagger e^{-ik}$.
2. A zero-energy solution $\psi_n=\lambda^n\varphi$ with $\lambda=e^{ik}$ (complex $k$) requires
   $$\big(H_1\lambda^2+H_0\lambda+H_1^\dagger\big)\varphi=0 .$$
3. With $\chi=\lambda\varphi$ this quadratic eigenproblem becomes the linear generalised one
   $$\begin{pmatrix}0&1\\-H_1^\dagger&-H_0\end{pmatrix}\!\begin{pmatrix}\varphi\\\chi\end{pmatrix}
   =\lambda\begin{pmatrix}1&0\\0&H_1\end{pmatrix}\!\begin{pmatrix}\varphi\\\chi\end{pmatrix}.$$
4. Decaying modes have $|\lambda|<1$. The slowest one sets the Majorana tail
   $|\psi|\propto e^{-x/\xi_M}$:
   $$\xi_M=-\frac{a}{\ln|\lambda|},\qquad k_\mathrm{osc}=\frac{|\arg\lambda|}{a}.$$

The code's lead fallback [leads.py:123-135](my_functions/leads.py#L123-L135) uses the same modes at finite $E$.

**Getting $H_0$ and $H_1$ from three Bloch matrices [derived here].**
- $H(0)=H_0+H_1+H_1^\dagger$, $H(\pi)=H_0-H_1-H_1^\dagger$ and $H(\pi/2)=H_0+i(H_1-H_1^\dagger)$.
- Hence $H_0=\tfrac12[H(0)+H(\pi)]$ and $H_1=\tfrac12\big[\tfrac12(H(0)-H(\pi))-i(H(\pi/2)-H_0)\big]$.

**Code**
- the block extraction: [study_oscillation.py:61-66](computations/study_oscillation.py#L61-L66);
- the eigenproblem: [:69-77](computations/study_oscillation.py#L69-L77);
- $\xi_M$, $k_\mathrm{osc}$: [:80-87](computations/study_oscillation.py#L80-L87);
- mixed SOC, 3 slowest modes: [study_mixed.py:62](computations/study_mixed.py#L62).

**Physics.** The tail is a superposition of a few complex-$k$ modes. The slowest one dominates at large distance;
its $\mathrm{Re}\,k$ follows the outer Fermi point (CLAUDE.md, oscillation section).

### 7.2 The gap dip: $\xi_M$ from the real band
**[derived here]**
1. Near the outer Fermi point $k_0$ the lowest band is a hyperbola,
   $$E(k)^2\simeq\Delta_\mathrm{out}^2+(\hbar v_A)^2(k-k_0)^2,\qquad \hbar v_A=\sqrt{\Delta_\mathrm{out}E''(k_0)}.$$
2. Continue it to $E=0$: $k=k_0\pm i\Delta_\mathrm{out}/\hbar v_A$, so
   $$\xi_\mathrm{hyp}=\frac{\hbar v_A}{\Delta_\mathrm{out}}.$$
3. This is the BCS form $\xi=\hbar v/\Delta$ with the Andreev-band velocity $v_A$, not $v_F$.
4. Beyond the hyperbola, $E^2$ is fitted by a degree-6 polynomial over $k_0\pm2/\xi_\mathrm{hyp}$ and continued to
   its nearest complex zero ($\xi_\mathrm{poly}$).

**Code:** [study_oscillation.py:90-115](computations/study_oscillation.py#L90-L115).

**Physics.** $\xi_M$ is a property of the narrow dip alone: a slow Andreev band (small $v_A$) plus a small gap at
$k_0$. That is why $\hbar v_F/\Delta_\mathrm{top}$ fails (CLAUDE.md).

### 7.3 The closed box: $\varepsilon$ exactly
**[derived here]** A finite junction of $N=L/a$ slices with hard walls at both ends, and no probes:

$$
H_\mathrm{box}=\mathbb 1_N\otimes H_0+S\otimes H_1+S^\mathsf T\otimes H_1^\dagger ,
\qquad S=\text{superdiagonal shift}.
$$

$\varepsilon$ is the smallest $|E|$, from sparse shift-invert at $\sigma=0$. It is the splitting $E_M$ of
Nilsson's Eq. (1); the BdG levels come as $\pm E_M$.
- **Code:** [study_oscillation.py:118-127](computations/study_oscillation.py#L118-L127).
- Cross-check: the lowest end-LDOS peak in the transport geometry, `splitting`
  [tools.py:548-553](computations/tools.py#L548-L553), gives 4.0 µeV vs the box's 4.09 µeV (CLAUDE.md).

### 7.4 The overlap law $\varepsilon\approx A\Delta e^{-L/\xi_M}|\cos(k_\mathrm{osc}L+\varphi_0)|$
**[source, verified]** Das Sarma, Sau & Stanescu, PRB 86, 220506(R) (2012), arXiv:1211.0539:
- **Eq. (1):** $\Delta E\sim\mathrm{Re}[\Psi_l^\dagger(L/2)\hat J\Psi_r(L/2)]$.
- **Eq. (2):** $\Psi_l\propto e^{-x/\xi}e^{\pm ik_{F,\mathrm{eff}}x}$.
- **Eq. (3):** $\Delta E\approx\hbar^2k_{F,\mathrm{eff}}\frac{e^{-2L/\xi}}{m\xi}\cos(k_{F,\mathrm{eff}}L)$.

**[derived here]** The exponent depends on the convention. Here $\xi_M$ is defined by $|\psi|\propto e^{-x/\xi_M}$
(§7.1), and the two ends are a distance $L$ apart. The overlap of Eq. (1) then scales as $e^{-L/\xi_M}$ and
oscillates as $\cos(k_\mathrm{osc}L+\varphi_0)$.
- The prefactor is not derived. $A$ and $\varphi_0$ are fitted, two parameters for all fields:
  [study_oscillation.py:254-263](computations/study_oscillation.py#L254-L263).
- The fit is a **test**: zeros spaced by $\pi$ in $k_\mathrm{osc}L$, a constant $A$.
- For the mixed SOC the phase holds but $A$ drifts (CLAUDE.md, "THE PHASE HOLDS, THE AMPLITUDE DOES NOT").

### 7.5 End modes and Rayleigh–Ritz (study_mixed [amplitude])
**[derived here]**
1. **Left end mode.** $\psi_L(n)=\sum_{|\lambda_j|<1}c_j\varphi_j\lambda_j^{\,n+1}$, a zero-energy solution in the
   bulk by §7.1. The hard wall $\psi_L(-1)=0$ means $\sum_jc_j\varphi_j=0$, so $c$ is the null vector of
   $[\varphi_j]$ (taken from the SVD).
2. **Right end mode.** The same with the growing modes: $\psi_R(n)=\sum_{|\lambda_j|>1}d_j\varphi_j\lambda_j^{\,n-N}$.
3. **Rayleigh–Ritz.** In $\mathrm{span}\{\psi_L,\psi_R\}$ solve $H_\mathrm{RR}u=E\,S_\mathrm{RR}u$, which gives
   $\varepsilon_\mathrm{RR}$.
4. **Boundary term.** $H\psi_L=0$ inside, so truncating at $N$ leaves only the missing neighbour:
   $$\varepsilon\simeq\frac{|\psi_R(N-1)^\dagger H_1\psi_L(N)|}{\|\psi_L\|\,\|\psi_R\|}.$$
5. **Semi-infinite norms** in closed form:
   $$\sum_{m\ge1}\Big|\sum_jc_j\varphi_jz_j^m\Big|^2=\sum_{jk}c_j^*c_k(\varphi_j^\dagger\varphi_k)\frac{z_j^*z_k}{1-z_j^*z_k}.$$

**Code**
- end modes: [study_mixed.py:477-490](computations/study_mixed.py#L477-L490);
- $\psi_L$, $\psi_R$: [:493-507](computations/study_mixed.py#L493-L507);
- norms: [:510-516](computations/study_mixed.py#L510-L516);
- RR and boundary term: [:525](computations/study_mixed.py#L525).

**Result:** $\varepsilon_\mathrm{RR}/\varepsilon_\mathrm{box}=0.9999$ wherever $L/\xi_M\ge3$. The lobe heights are
the overlap of the two end Majoranas.

### 7.6 LDOS and total DOS
- **[source, verified]** Gresta et al., **Eq. (8)**: $\mathrm{LDOS}=\frac1\pi\mathrm{Im\,Tr}\,\mathcal G^a$. That is
  equivalent to $-\frac1\pi\mathrm{Im\,Tr}\,G^r$, the form of Lewenkopf–Mucciolo **Eq. (12)**, which the code uses
  ([spectra.py:149](my_functions/spectra.py#L149)).
- Total DOS **[derived here]**:
  $$\mathrm{Tr}\,G=\mathrm{Tr}\,g_0+\mathrm{Tr}\big(K^{-1}W(g_0^2)_{PP}W^\dagger\big),$$
  with $K$ as in §2.2.
  - $(g_0^2)$ is the off-diagonal block of $\begin{pmatrix}A&-1\\0&A\end{pmatrix}^{-1}$, which is still
    block-tridiagonal.
  - Code: [tools.py:478-539](computations/tools.py#L478-L539).

---

## 8. Index: quantity → formula → code → source

| quantity | formula | code | source |
|---|---|---|---|
| BdG model | Scharf Eq. (1), discretised | [hamiltonians.py:25](my_functions/hamiltonians.py#L25), [ldos.py:196](computations/ldos.py#L196) | Scharf 2019 Eq. (1); lattice: Mireles–Kirczenow Eqs. (18), (24)–(25) |
| $E_Z$, $E_T$ | $g\mu_BB/2$, $\frac\pi2\hbar v_F/W$ | [ldos.py:148](computations/ldos.py#L148), [:157](computations/ldos.py#L157) | Scharf, Sec. VII text; Eq. (8) |
| $\Sigma,\Gamma$ | $V_c^\dagger g_sV_c$, $i(\Sigma-\Sigma^\dagger)$ | [leads.py:239](my_functions/leads.py#L239), [rgf.py:293](my_functions/rgf.py#L293) | Lewenkopf–Mucciolo Eqs. (16)–(17) |
| $g_s$ (lead surface) | $\varepsilon_s=H_0+V(z-\varepsilon_s)^{-1}V^\dagger$, decimation | [leads.py:48](my_functions/leads.py#L48) | LM Eqs. (60)–(61), (86)–(93) (typo in (91)) |
| $G^r$ (centre) | RGF plus ribbon Dyson step | [rgf.py:115](my_functions/rgf.py#L115), [:342](my_functions/rgf.py#L342) | LM Eqs. (19)–(20), (32), (35), (38)–(39); Dyson step follows from LM Eq. (19) |
| channels | $\mathrm{Tr}[\Gamma G^r\Gamma G^a]$ projected on e/h | [rgf.py:373](my_functions/rgf.py#L373) | Gresta Eq. (7) |
| $G$ | $\mathcal T^{ee}-\mathcal T^{he}$ | [tools.py:290](computations/tools.py#L290) | Gresta Eqs. (3), (5) |
| $\kappa$ | $\mathcal T^{ee}+\mathcal T^{he}$; kernel $\frac{3}{\pi^2}x^2W_\mathrm{el}$ | [tools.py:290](computations/tools.py#L290), [transport.py:118](my_functions/transport.py#L118) | Gresta Eqs. (4), (6); Klees Sec. IV and App. C |
| local Andreev | $R_{eh}+R_{he}$ | [tools.py:290](computations/tools.py#L290) | follows from Klees App. C; Flensberg Eq. (13); Law–Lee–Ng |
| $\det r$ | $-1+i\Gamma^{1/2}G_{LL}\Gamma^{1/2}$ | [tools.py:266](computations/tools.py#L266) | LM Eqs. (5), (8); Fulga Table I; Akhmerov Eq. (3) |
| two-Majorana $\kappa$, $\det r$ | $\frac{2x^2}{(1+x^2)^2}$, $\frac{x^2-1}{x^2+1}$ | [study_tuning.py:458](computations/study_tuning.py#L458) | follows from Nilsson Eqs. (1)–(6) (limit stated after their Eq. (8)); $\kappa(\det r)$ also from Akhmerov Eq. (3) |
| $\Gamma$ (direct) | HWHM of $R_{he}(E)$ | [study_mixed.py:88](computations/study_mixed.py#L88) | follows from Nilsson Eqs. (4)–(6); Flensberg Eq. (13) |
| $Q$ | $\mathrm{sgn\,Pf}\tilde B(0)\,\mathrm{sgn\,Pf}\tilde B(\pi)$ | [tools.py:177](computations/tools.py#L177) | Kitaev Eq. (26); Wimmer (Parlett–Reid) |
| $\xi_M$, $k_\mathrm{osc}$ | $-a/\ln\lvert\lambda\rvert$, $\lvert\arg\lambda\rvert/a$ | [study_oscillation.py:69](computations/study_oscillation.py#L69) | modes: Kwant Eqs. (5)–(6); $\xi_M$, $k_\mathrm{osc}$ derived here |
| $\xi_\mathrm{hyp}$ | $\hbar v_A/\Delta_\mathrm{out}$ | [study_oscillation.py:90](computations/study_oscillation.py#L90) | derived here |
| $\varepsilon$ | lowest $\lvert E\rvert$ of the closed box | [study_oscillation.py:118](computations/study_oscillation.py#L118) | (= Nilsson's $E_M$) |
| overlap law | $A\Delta e^{-L/\xi_M}\lvert\cos(kL+\varphi_0)\rvert$ | [study_oscillation.py:254](computations/study_oscillation.py#L254) | Das Sarma et al. Eqs. (1)–(3); exponent convention derived here |
| $\varepsilon_\mathrm{RR}$ | Rayleigh–Ritz in $\{\psi_L,\psi_R\}$ | [study_mixed.py:525](computations/study_mixed.py#L525) | derived here |
| LDOS | $-\frac1\pi\mathrm{Im\,Tr}\,G^r$ | [spectra.py:149](my_functions/spectra.py#L149) | Gresta Eq. (8); LM Eq. (12) |

## 9. Sources
All papers were checked against their arXiv LaTeX source; equation numbers were checked against the printed PDF.

1. B. Scharf, F. Pientka, H. Ren, A. Yacoby, E. M. Hankiewicz, *Tuning topological superconductivity in
   phase-controlled Josephson junctions with Rashba and Dresselhaus spin-orbit coupling*, PRB 99, 214503 (2019),
   arXiv:1904.08981. Used: Eq. (1), Eq. (8), $E_Z=g\mu_BB/2$ (Sec. VII), class D via Kitaev (Sec. III).
2. D. Gresta, F. Domínguez, R. L. Klees, F. Goth et al., *Limits of thermal conductance quantization in chiral
   topological Josephson junctions*, PRB 114, 125405 (2026), arXiv:2602.12947. Used: Eqs. (3)–(8), $\kappa_0$, the
   half-quantum statement.
3. R. L. Klees, D. Gresta, J. Sturm, L. W. Molenkamp et al., *Majorana-mediated thermoelectric transport in
   multiterminal junctions*, PRB 110, 064517 (2024), arXiv:2306.17845. Used: App. C (charge and heat currents),
   Sec. III (particle–hole symmetry of $\mathcal T$), Sec. IV (Onsager coefficients, $\mathcal T_{22}$).
4. C. H. Lewenkopf, E. R. Mucciolo, *The recursive Green's function method for graphene*, J. Comput. Electron. 12,
   203 (2013), arXiv:1304.3934. Used: Eqs. (5), (8), (12), (15)–(20), (32), (35), (38)–(39) (Sec. 3, RGF),
   (60)–(61) (Sec. 4.1, lead self-consistency), (86)–(93) (Sec. 4.3, decimation; Eq. (91) has a typo, §2.1).
5. M. P. López Sancho, J. M. López Sancho, J. Rubio, J. Phys. F 15, 851 (1985). Decimation. Not checked in its own
   text; it is cited as in refs. 2 and 4, and the recursion was checked as printed in ref. 4.
6. I. C. Fulga, F. Hassler, A. R. Akhmerov, C. W. J. Beenakker, *Scattering formula for the topological quantum
   number of a disordered multi-mode wire*, PRB 83, 155429 (2011), arXiv:1101.1749. Used: Table I, Sec. III.
7. J. Nilsson, A. R. Akhmerov, C. W. J. Beenakker, *Splitting of a Cooper pair by a pair of Majorana bound states*,
   PRL 101, 120403 (2008), arXiv:0806.4889. Used: Eqs. (1)–(8).
8. C. W. J. Beenakker, *Random-matrix theory of Majorana fermions and topological superconductors*, RMP 87, 1037
   (2015), arXiv:1407.2131. Used: the Cayley/Mahaux–Weidenmüller form, $S(0)$ real orthogonal.
9. K. T. Law, P. A. Lee, T. K. Ng, *Majorana fermion induced resonant Andreev reflection*, PRL 103, 237001 (2009),
   arXiv:0907.1909. Used: $2e^2/h$.
10. A. Kitaev, *Unpaired Majorana fermions in quantum wires*, Phys.-Usp. 44, 131 (2001), arXiv:cond-mat/0010440.
    Used: Eq. (26).
11. M. Wimmer, *Efficient numerical computation of the Pfaffian for dense and banded skew-symmetric matrices*,
    ACM TOMS 38, 30 (2012), arXiv:1102.3440. Used: the Parlett–Reid algorithm.
12. S. Das Sarma, J. D. Sau, T. D. Stanescu, *A Majorana smoking gun for the superconductor-semiconductor hybrid
    topological system* (arXiv title), PRB 86, 220506(R) (2012), arXiv:1211.0539. Used: Eqs. (1)–(3).
13. F. Pientka, A. Keselman, E. Berg, A. Yacoby, A. Stern, B. I. Halperin, *Topological superconductivity in a
    planar Josephson junction*, PRX 7, 021032 (2017), arXiv:1609.09482. The platform, as cited by Scharf.
14. C. Caroli, R. Combescot, P. Nozières, D. Saint-James, J. Phys. C 4, 916 (1971). The Caroli formula, as cited
    by refs. 2 and 4.
15. A. R. Akhmerov, J. P. Dahlhaus, F. Hassler, M. Wimmer, C. W. J. Beenakker, *Quantized conductance at the
    Majorana phase transition in a disordered superconducting wire*, PRL 106, 057001 (2011), arXiv:1009.5542.
    Used: Eq. (3) and the text after it ($\det r=\prod\tanh\lambda_n$, $T_n=1/\cosh^2\lambda_n$, $G_\mathrm{th}$).
16. K. Flensberg, *Tunneling characteristics of a chain of Majorana bound states*, PRB 82, 180516 (2010),
    arXiv:1009.3533. Used: Eqs. (3), (9), (13) (tunnel coupling, $\Gamma$, Andreev Lorentzian).
17. C. W. Groth, M. Wimmer, A. R. Akhmerov, X. Waintal, *Kwant: a software package for quantum transport*, New J.
    Phys. 16, 063065 (2014), arXiv:1309.2926. Used: Sec. 2.2, Eqs. (5)–(6) (lead modes, complex $\lambda$).
18. F. Mireles, G. Kirczenow, *Ballistic spin-polarized transport and Rashba spin precession in semiconductor
    nanowires*, PRB 64, 024426 (2001), arXiv:cond-mat/0105049. Used: Sec. II.C, Eqs. (18), (24)–(25)
    (tight-binding Rashba term, $t_\mathrm{so}=\alpha/2a$).

The journal references of refs. 15, 16 and 18 were checked against Crossref.
