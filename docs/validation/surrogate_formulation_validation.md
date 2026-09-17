# Validation of the Empirical Aerodynamic/Structural Surrogate Against Published Literature

**Scope:** `src/alphadesign/cfd_analysis.py` (`enhanced_airfoil_lift_coefficient`, `enhanced_airfoil_drag_coefficient`, `calculate_ground_effect`, `calculate_slot_effect`, `calculate_reynolds_number`, `calculate_dynamic_pressure`) and `src/alphadesign/formula_constraints.py` (`natural_frequency`, `critical_buckling_stress`, `safety_factor_static`, `safety_factor_fatigue`, and the `efficiency_computed = total_downforce / total_drag` metric).

**Method:** Every formula was compared against primary or authoritative secondary literature. Where possible, sources were fetched and read directly (NASA Glenn *Beginner's Guide to Aeronautics* pages, peer-reviewed paper records with DOIs verified via Crossref/OpenAlex, university lecture notes, and full-text book search via the Internet Archive / Open Library "search inside" index). Where a formula could not be traced to any published equation, this is stated plainly. All URLs and DOIs listed in the bibliography were actually retrieved during this review.

## Executive summary

Of the 13 formulas reviewed, **5 are TEXTBOOK_EXACT** (induced drag with Oswald factor, cantilever first natural frequency, the safety-factor definitions, dynamic pressure, and the downforce/drag efficiency metric), **4 are TEXTBOOK_GENERAL_FORM_TUNED_COEFFICIENTS** (lift-curve slope with Prandtl–Glauert and thickness, zero-lift angle from camber, the Reynolds-number drag correction, and plate buckling), **2 are PLAUSIBLE_ENGINEERING_APPROXIMATIONS** (zero-lift drag buildup and the slot/gap effect), and **2 are QUESTIONABLE** (both ground-effect factors, which contain piecewise discontinuities at their branch points and contradict measured force-reduction behaviour at very low ride height). The honest overall conclusion: the surrogate is *anchored* in real theory — its skeleton is thin-airfoil theory, Prandtl lifting-line theory, Prandtl–Glauert compressibility, Euler–Bernoulli beam vibration, and classical plate buckling — but nearly every numerical coefficient beyond the textbook constants is a project-specific fit, and the two ground-effect functions have internal discontinuities plus a qualitatively wrong trend in the low-ride-height regime that F1 front wings actually operate in. The paper should describe this surrogate as a *physics-informed heuristic model*, validate it against CFD or wind-tunnel data before making quantitative claims, and add the limitations listed per formula below.

---

## 1. Lift-curve slope with compressibility + thickness correction

`cl_alpha = (2*pi/beta) * (1 + 0.77*thickness)`, `beta = sqrt(1 - mach^2)`

**What it claims to model.** The 2D section lift-curve slope of each wing element: thin-airfoil value 2π per radian, amplified by the Prandtl–Glauert compressibility factor 1/β and by a linear thickness correction.

**Literature basis.**
- The inviscid thin-airfoil slope of exactly 2π per radian is the classical result of thin-airfoil theory (see e.g. the Stanford AA200b thin-airfoil lecture notes, which also note viscous effects make real slopes lower) [B19]; Anderson, *Fundamentals of Aerodynamics*, Ch. 4 [B24].
- The Prandtl–Glauert correction Cl_α = 2π/√(1−M²) is the standard linearized-compressibility result (Wikipedia, "Prandtl–Glauert transformation" [B4]; derivation summarised in [B20]).
- That 2D lift-curve slope *increases slightly with airfoil thickness* in inviscid flow is documented [B20, a/14559]. However, the specific coefficient **0.77** could not be traced to any published source despite targeted full-text searches of aerodynamics textbooks ("1+0.77", "0.77 t/c" return nothing). It appears to be a project fit. Note also that inviscid theory raises the slope with thickness while *viscous* effects lower it below 2π (explicitly stated in Thomas, *Fundamentals of Sailplane Design* [B27]); the code applies the former and ignores the latter, so at t/c = 0.12, M = 0.1 it predicts cl_α ≈ 6.90 /rad (0.120 /deg), roughly 15–20% above the ~0.10–0.11 /deg typically measured for real sections at F1-relevant Reynolds numbers.

**Assessment: TEXTBOOK_GENERAL_FORM_TUNED_COEFFICIENTS.** The 2π/β part is textbook; the `(1 + 0.77·t/c)` factor has the right sign and rough magnitude of the inviscid thickness effect but the constant 0.77 is unverifiable, and the net number overshoots measured section slopes.

**Recommendation.** Cite the Prandtl–Glauert/thin-airfoil basis and state that the thickness coefficient is an empirical fit. Flag the systematic overprediction of section slope (no viscous slope reduction), or add a viscous efficiency factor (~0.9) as done in e.g. DATCOM-style methods. Also note the `beta = 0.1` clamp at Mach ≥ 0.9 is an arbitrary guard, irrelevant at F1 speeds.

---

## 2. Zero-lift angle from camber

`alpha_0 = -2*camber*(1 + 0.5*thickness)` (camber, thickness as fractions; result in radians)

**What it claims to model.** The zero-lift angle of a cambered element, proportional to camber, with a thickness modifier.

**Literature basis.** Thin-airfoil theory for a *parabolic* camber line z(x) = 4εx(1 − x/c) gives exactly α_L0 = −2ε (radians) — this is the standard worked example of cambered thin-airfoil theory (Pope, *Basic Wing and Airfoil Theory*, 1951, Ch. 7 §2 "The Parabolic-camber Airfoil" [B25]; Anderson, *Fundamentals of Aerodynamics*, 6th ed., §4.8 cambered-airfoil thin-airfoil theory [B24]; the parabolic mean line is the example case in the Stanford AA200b notes [B19]). I independently re-derived the result: with dz/dx = 4ε·cosθ, α_L0 = −(1/π)∫₀^π (dz/dx)(cosθ−1)dθ = −2ε. The `(1 + 0.5·t/c)` thickness multiplier is **not** part of thin-airfoil theory (inviscid thickness has no first-order effect on α_L0) and could not be traced to any source — it is a tuning factor. For other camber families (e.g. NACA 4-digit) the coefficient of camber differs slightly from 2.0, and measured α_L0 differs from theory, so −2ε is itself an approximation for real F1 profiles.

**Assessment: TEXTBOOK_GENERAL_FORM_TUNED_COEFFICIENTS.** The −2·camber core is an exact thin-airfoil result for parabolic camber; the thickness modifier is project tuning.

**Recommendation.** Cite thin-airfoil theory (parabolic camber) for the −2ε term; state the thickness term is empirical. Sanity check is favourable: camber 2%, t/c 12% → α_0 = −0.042 rad ≈ −2.4°, in the right ballpark for a NACA xx12-type section.

---

## 3. Zero-lift (profile) drag buildup

`cd_0 = 0.006 + 0.02*camber + 0.05*thickness^2`

**What it claims to model.** The section minimum (zero-lift) profile drag as a baseline plus penalties growing with camber and with thickness squared.

**Literature basis.** No published closed-form equation of this form was found. The physics decomposition (profile drag = skin-friction drag + form drag, both of which grow with thickness and camber) is standard [B12]. The baseline constant 0.006 is realistic: measured minimum profile-drag coefficients of smooth conventional sections at Re ~ 3–6·10⁶ are in the ~0.005–0.008 range (NACA Report 824, "Summary of Airfoil Data", Abbott, von Doenhoff & Stivers, 1945 — the canonical data compendium behind Abbott & von Doenhoff, *Theory of Wing Sections*, Dover 1959 [B28, B23]). The specific linear-camber / quadratic-thickness penalties (0.02, 0.05) appear fitted: e.g. t/c = 0.12 adds only 0.00072, whereas established form-factor methods grow form drag roughly linearly in t/c. Camber's effect on minimum drag in reality is small and depends on where the drag bucket sits, not a monotonic penalty.

**Assessment: PLAUSIBLE_ENGINEERING_APPROXIMATION.** Magnitudes are realistic and the trend directions are defensible, but the functional form and coefficients are not from any identifiable source.

**Recommendation.** Do not present this as a literature equation. Either cite it as a project fit anchored to NACA R-824-scale minimum-drag values, or replace with a skin-friction × form-factor buildup (DATCOM/Raymer-style) which has a published pedigree. Flag that the monotonic camber penalty misrepresents the drag-bucket behaviour of real sections.

---

## 4. Reynolds-number drag correction

`re_factor = (Re/1e6)^(-0.15)` for Re > 1e6, else `1.2`

**What it claims to model.** Decay of profile drag with increasing Reynolds number, with a flat 20% low-Re penalty below 10⁶.

**Literature basis.** Power-law decay of friction drag with Re is textbook boundary-layer theory: for a turbulent flat plate, the local skin-friction coefficient scales as Re_x^(−1/5) (Prandtl's one-seventh-power law: cf = 0.0576·Re_x^(−1/5), verified on Wikipedia "Skin friction drag" [B7]); the more accurate Prandtl–Schlichting relation is cf = 0.455/(log₁₀Re)^2.58 (Schlichting & Gersten, *Boundary-Layer Theory* [B22]). The code's exponent −0.15 has the right sign and form but is neither −1/5 nor the log-law; it appears tuned, and it is applied to *total* profile drag, not just the friction component. The flat 1.2 penalty for Re ≤ 10⁶ captures the real trend of rising section drag at low Re, but as a constant it is arbitrary.

**Internal issue:** the two branches are discontinuous at Re = 10⁶ — the factor jumps from 1.2 (just below) to 1.0 (just above), an abrupt ~17% drop in profile drag. An F1 front-wing main element at 150–350 km/h with c ≈ 0.2–0.4 m sits at Re ≈ 1–2·10⁶, i.e. right at the breakpoint, so optimiser designs can exploit the step.

**Assessment: TEXTBOOK_GENERAL_FORM_TUNED_COEFFICIENTS.** Recognised power-law form with a tuned exponent and a discontinuity at the branch point.

**Recommendation.** Use a smooth blend across Re = 10⁶ (e.g. continue the power law through the breakpoint or use Prandtl–Schlichting). State the −0.15 exponent as a fit. Flag the discontinuity explicitly in the limitations section.

---

## 5. Induced drag with Oswald efficiency

`cd_induced = cl^2 / (pi * AR * e)`, `e = 0.7 - 0.05*element_index`

**What it claims to model.** Drag-due-to-lift of each element via Prandtl's finite-wing result, with the Oswald span efficiency degraded for flap elements.

**Literature basis.** The equation is exactly the Prandtl lifting-line result: NASA Glenn's *Beginner's Guide to Aeronautics* gives it verbatim, Cdi = Cl²/(π·AR·e) [B1]; see also Anderson, *Fundamentals of Aerodynamics*, Ch. 5 [B24]. e = 0.7 for a main element is a plausible Oswald value (real wings ~0.7–0.9); degrading e by 0.05 per flap index is a heuristic but reasonable in direction (deflected multi-element systems are farther from elliptic loading).

**Implementation caveat (not the formula itself):** the per-element aspect ratio is computed as `element_span = sqrt(element_area / chord)` then `AR = element_span / chord`. Dimensionally, span = area/chord; the `sqrt` is dimensionally wrong (units of √m) and distorts the AR (e.g. a 1.8 m span, 0.3 m chord element gets AR ≈ 4.5 instead of 6.0). This looks like a bug: `element_span = element_area / chord` was presumably intended.

**Assessment: TEXTBOOK_EXACT** (the formula is Prandtl's equation verbatim; e is a parameter choice, and 0.7 is within the accepted range).

**Recommendation.** Fix or justify the `sqrt` in the element-AR computation. Cite Prandtl lifting-line theory; state that e = 0.7 − 0.05·i is a heuristic assignment. Note that per-element induced drag summed independently double-counts spanwise interference of a continuous multi-element wing — a limitation for quantitative accuracy.

---

## 6. Ground-effect factor, main element

> **Fixed 2026-09-17.** The formula below is the original, as reviewed. It has since been replaced
> with a curve fit by least squares to real data from J. Zerihan's PhD thesis (Univ. of
> Southampton, 2001); see [`docs/validation/README.md`](README.md#fix-ground-effect-formula) and
> [`ground_effect_source_data.md`](ground_effect_source_data.md) for the fit and full source data.
> The analysis below is kept as the historical record of what was wrong and why.

`h/c < 0.1: 2.2 − 1.2*(h/c)`; `0.1 ≤ h/c < 0.5: 1 + 1.2*exp(−3*h/c)`; `h/c ≥ 0.5: 1 + 0.2*exp(−h/c)`; capped at 2.5

**What it claims to model.** A multiplier applied directly to the element's lift coefficient (`cl_element *= ground_effect * slot_effect`) representing downforce gain of the main wing in ground proximity as a function of ride height over chord.

**Literature basis.** The canonical experimental studies are Zerihan & Zhang, "Aerodynamics of a Single Element Wing in Ground Effect," *Journal of Aircraft* 37(6):1058–1064, 2000 [B13] and the review by Zhang, Toet & Zerihan, "Ground Effect Aerodynamics of Race Cars," *Applied Mechanics Reviews* 59(1):33–49, 2006 [B16]. The measured trend — downforce rises significantly for h/c below ~0.2 — matches the code's general shape, and the piecewise exponentials look hand-fitted to such curves. **However**, [B13] measured a clear **force-reduction phenomenon**: below h/c ≈ 0.08–0.1 the wing enters trailing-edge separation and downforce *drops*. The code instead rises monotonically to its cap (2.2 at h/c → 0), i.e. it claims maximum downforce exactly where experiment shows stall-driven loss. This is precisely the regime F1 front-wing mainplanes operate in.

**Internal issue — discontinuities at both branch points.** At h/c = 0.1 the factor jumps from 2.08 to 1.89 (−9%); at h/c = 0.5 from 1.268 to 1.121 (−12%). No physical justification; an optimiser will see spurious gradients there.

**Assessment: QUESTIONABLE.** Qualitative trend correct over h/c ∈ [0.1, ~0.5], but (a) the formula contradicts published measurements in the h/c < 0.1 force-reduction regime, and (b) it contains non-physical discontinuities at both breakpoints.

**Recommendation.** Highest-priority fix. Fit a smooth single function (no branch points) that reproduces the measured downforce-vs-h/c curve *including* the turnover below h/c ≈ 0.1 (e.g. a rise-and-fall curve anchored to [B13]/[B16] data). Until then, the paper must state that the surrogate cannot represent low-ride-height stall and will overpredict downforce there, and that branch-point jumps create artificial optimiser incentives.

---

## 7. Ground-effect factor, flap elements

> **Fixed (structurally) 2026-09-17.** No longer discontinuous and never falls below 1.0. The
> amplitude is still a heuristic, not a data fit: see
> [`docs/validation/README.md`](README.md#fix-ground-effect-formula). The analysis below is kept
> as the historical record of what was wrong and why.

Same piecewise shape, scaled by `0.8^element_index`; breakpoints 0.15 / 0.8; third branch `1 + 0.1*exp(−h/c)*0.8^i`

**What it claims to model.** The same ground-proximity downforce multiplier for flap elements, attenuated because flaps sit higher above the ground and shielded by the main element.

**Literature basis.** The double-element measurements are Zhang & Zerihan, "Aerodynamics of a Double-Element Wing in Ground Effect," *AIAA Journal* 41(6):1007–1016, 2003 [B14]: total downforce rises with proximity (with a plateau or partial reduction depending on flap angle), reaches a maximum, then reduces near the ground; the main element generates most of the downforce. That flaps benefit *less* from ground effect than the main element is qualitatively supported, but the specific `0.8^i` attenuation law, breakpoints, and coefficients are not from any identified source.

**Internal issues.** (a) Discontinuities at both breakpoints (e.g. for element 1: 1.344 → 1.274 at h/c = 0.15; and an *upward* jump 0.929 → 1.036 at h/c = 0.8, a sign-inconsistent step). (b) Because `0.8^i` multiplies the *whole* factor in the first two branches, the factor falls **below 1.0** over part of the range (element 1: ≈ 0.93 for h/c ∈ [0.35, 0.8); element 2: lower still) — i.e. the model predicts ground proximity *reduces* flap downforce below its freestream value, while branch 3 asymptotes to exactly 1.0. Nothing in [B14] supports a below-unity region; at most the flap gain is weaker. (c) No force-reduction turnover at very low h/c, same defect as #6.

**Assessment: QUESTIONABLE.** Weaker-gain direction matches [B14] qualitatively, but the piecewise discontinuities (one of which jumps the *wrong way*), the below-unity region, and the missing low-height turnover are inconsistent with the measurements.

**Recommendation.** Same as #6: replace with a smooth fit to [B14] flap-element data, keep factors ≥ 1 (asymptote to 1 from above), include the force-reduction regime, and document the attenuation as a fit.

---

## 8. Multi-element slot/gap effect

`gap_efficiency = exp(−((gap_ratio − 0.02)/0.01)^2)`; `circulation_boost = 1.3 + 0.15*gap_efficiency*overlap_efficiency`; `velocity_ratio = 1.4 + 0.4*gap_efficiency`; `cl_multiplier = circulation_boost * sqrt(velocity_ratio)`

**What it claims to model.** The lift benefit a flap element receives from the main-element/flap slot: a Gaussian "efficiency" centred on an optimal gap of 2% chord, modulating circulation and slot-velocity boosts whose product (with a square root on velocity) multiplies the flap's cl.

**Literature basis.** That the slot gap between elements has an optimum and strongly controls multi-element high-lift performance is well established experimentally: Smith, "High-Lift Aerodynamics," *Journal of Aircraft* 12(6):501–530, 1975 (the 37th Wright Brothers Lecture — the canonical reference on slat/flap slot physics: fresh boundary layer, circulation, dumping and off-surface pressure-recovery effects) [B15]; Lin & Dominik, "Parametric Investigation of a High-Lift Airfoil at High Reynolds Numbers," *Journal of Aircraft* 34(4):485–491, 1997 (gap/rigging parametrics) [B17]; Zhang & Zerihan 2003 study gap flow specifically for a race-car-style double-element wing [B14]. But the *specific equations* — a Gaussian in gap ratio centred at exactly 2% with 1% width, a circulation boost of 1.3 + 0.15·eff, and especially `cl_multiplier = circulation_boost · sqrt(velocity_ratio)` — have no published counterpart I could find. The square root of a velocity ratio multiplying a lift coefficient is dimensionally arbitrary (lift already scales with v²). Note also the modulation is shallow: the multiplier ranges only from ≈ 1.54 (far off-optimum) to ≈ 1.95 (optimum), so the "optimal gap" hardly matters in the model.

**Assessment: PLAUSIBLE_ENGINEERING_APPROXIMATION.** Directionally consistent with slot-flow physics (optimum gap exists, slot flow re-energises the flap boundary layer), but the equations themselves are constructed heuristics.

**Recommendation.** Present as a project heuristic motivated by Smith's slot-effect physics [B15]; do not call it "physics-based" (the code's comment claims this). Calibrate the optimum gap and multiplier magnitudes against multi-element data (e.g. [B14], [B17]) or CFD. Flag the weak sensitivity to gap and the arbitrary sqrt in the limitations.

---

## 9. Cantilever beam first natural frequency

`f = (1.875^2 / (2*pi)) * sqrt(E*I / (mass_per_length * L^4))`

**What it claims to model.** The first bending natural frequency of the wing modelled as a uniform cantilever (fixed-free) Euler–Bernoulli beam.

**Literature basis.** This is exactly the classical result: for a uniform cantilever, ω₁ = (β₁L)²·√(EI/(μL⁴)) with β₁L = 1.8751 the first root of cos(βL)·cosh(βL) = −1, and f = ω/2π. The general relation βₙ := (μωₙ²/EI)^(1/4) and the cantilever boundary conditions are given in Wikipedia's "Euler–Bernoulli beam theory" article [B5]; the tabulated root 1.8751 appears in Blevins, *Formulas for Natural Frequency and Mode Shape* (1979, Table 8-1) [B21] — I additionally verified the root numerically (brentq on cosh(x)cos(x)+1 = 0 gives 1.87510407, matching the code's 1.875).

**Model caveats (the formula is right; the application is questionable):**
1. The code passes **L = total span**. A wing mounted at the fuselage centreline behaves as two cantilevers of length span/2; since f ∝ 1/L², using the full span **underestimates the frequency by a factor of 4** (a clamped–clamped beam of full span has β₁L = 4.730, i.e. ~6.4× the code value). Either way the code's number is biased low.
2. Boundary conditions are inconsistent within the same structural model: the bending moment uses a *simply supported* WL/8 while the frequency uses *cantilever* physics.
3. `E` is the longitudinal composite modulus; actual flexural rigidity of a laminate needs the D-matrix, and rotary inertia/shear (Timoshenko beam) matter for stubby composite wings. Tip masses (endplates) lower the frequency and are ignored.

**Assessment: TEXTBOOK_EXACT** for the equation itself.

**Recommendation.** The equation can be cited to Blevins/EB theory, but the paper must state and justify the effective length and boundary-condition assumptions (use span/2 for a centre-mounted wing, or a clamped–clamped model with β₁L = 4.730), reconcile the simply-supported statics with the cantilever dynamics, and note laminate-theory and tip-mass effects. The validation gate `natural_frequency > 30 Hz` inherits these errors.

---

## 10. Flat-plate critical buckling stress

`sigma_cr = (pi^2 * E * t^2) / (12 * (1 − 0.3^2) * b^2)` (t = wall thickness, b = section width = root chord)

**What it claims to model.** Local buckling of the wing skin as a flat plate under compressive bending stress, with Poisson's ratio hard-coded to 0.3.

**Literature basis.** This is the classical plate-buckling stress **with buckling coefficient k implicitly set to 1**: the standard form is σ_cr = k·π²E/(12(1−ν²))·(t/b)², derived from the plate stability equation (derived step-by-step in Wikipedia's "Buckling" article, which gives σ_cr = k_cr·π²E/(12(1−ν²)(b/t)²) with k_cr = (mb/a + a/mb)² for a simply supported plate [B6]; the authoritative textbook treatment is Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed. 1961, Ch. 7 "Buckling of Thin Plates" [B22-T]). The value k = 1 corresponds to the **wide-column limit** — a plate strip with both long (unloaded) edges free. For a plate simply supported on all four edges, k_cr = (mb/a + a/mb)² has minimum **k = 4** [B6]; for one long edge free (flange-like), the tabulated value is k ≈ 0.425 (Timoshenko & Gere; this specific number I did not independently verify online). So k = 1 sits between the two canonical cases and matches neither: it is 4× *unconservative* relative to a free-edged skin strip and 4× *conservative* relative to a fully supported panel. Two further issues: b is set to the **full root chord** (as if the skin had no rib/spar support within a chord width), and an isotropic `E` with ν = 0.3 is used for a **CFRP laminate**, for which orthotropic buckling analysis using the laminate D-matrix is the standard approach (Jones, *Mechanics of Composite Materials* [B29]).

**Assessment: TEXTBOOK_GENERAL_FORM_TUNED_COEFFICIENTS.** The functional form is the textbook plate-buckling equation; the implicit k = 1 is an unstated edge-condition assumption, and the isotropic substitution for a composite skin is a modelling approximation.

**Recommendation.** State the assumed k and edge conditions explicitly, or use k = 4 (panel supported by spars/ribs) / k = 0.425 (unsupported cover) as bounding cases. Use the true panel width between supports rather than the full chord, and replace E/(1−ν²) with the orthotropic plate rigidities for the CFRP laminate. Cite Timoshenko & Gere §plate buckling.

---

## 11. Static/fatigue safety factor vs von Mises stress

`safety_factor_static = ultimate_strength / von_mises_stress`; `safety_factor_fatigue = fatigue_limit / von_mises_stress`; with `von_mises = sqrt(stress_bending^2 + 3*shear^2)`

**What it claims to model.** Factor of safety as material allowable over computed equivalent stress, for both static ultimate and fatigue allowables.

**Literature basis.** The definition is the standard one: factor of safety = structural capability / applied load (or strength/stress) — Wikipedia "Factor of safety" [B9]; Shigley's *Mechanical Engineering Design* [B30]. The equivalent-stress form √(σ² + 3τ²) is exactly the von Mises distortion-energy equivalent for a biaxial bending-plus-shear stress state (the general criterion and the k = σ_y/√3 shear relation are given in Wikipedia "von Mises yield criterion" [B8]). **For an isotropic ductile metal this is the correct usage.** But the structure here is explicitly a **CFRP composite**, and von Mises is not valid for orthotropic/brittle materials with direction-dependent strengths: composite practice uses laminate-level criteria such as Tsai–Wu (Wikipedia "Tsai–Wu failure criterion" [B10]), Hashin, or max-stress against separate longitudinal tensile/compressive and shear allowables. The code partially acknowledges this (it computes a separate compression SF against bending stress) but the headline SF still divides the *tensile* ultimate by an isotropic von Mises stress. On the fatigue side, dividing a fatigue limit by a static-equivalent stress with no mean-stress correction (Goodman/Gerber), no cycle spectrum, and no stress-ratio dependence is a crude screening check, not a fatigue assessment.

**Assessment: TEXTBOOK_EXACT** for the safety-factor definition and the von Mises equivalent-stress algebra — with the explicit caveat that von Mises is the *wrong failure theory class* for an anisotropic composite structure.

**Recommendation.** Keep the SF definition; for the composite wing box, implement or cite a composite failure criterion (Tsai–Wu/Hashin/max-stress) with direction-specific allowables, and state that the current von Mises check is a conservative screening surrogate. Flag the fatigue check as limit-load screening only (no mean-stress or spectrum effects). This distinction must appear in the paper's limitations.

---

## 12. Dynamic pressure

`q = 0.5 * rho * V^2`

**What it claims to model.** Free-stream dynamic pressure used to convert coefficients to forces.

**Literature basis.** Textbook definition, derived from Bernoulli's equation: NASA Glenn's *Beginner's Guide to Aeronautics* "Dynamic Pressure" page derives p_s + ρu²/2 = p_t, i.e. q = ρV²/2 [B2]; used identically in the NASA lift/drag coefficient definitions [B3]. Valid for incompressible flow; at F1 speeds (≤ ~350 km/h, M ≈ 0.29) the incompressible form is fine as a first approximation (compressible correction ~2%).

**Assessment: TEXTBOOK_EXACT.**

**Recommendation.** None required. Optionally note the incompressible assumption at M ≈ 0.3.

---

## 13. Aerodynamic efficiency metric

`efficiency = total_downforce / total_drag`

**What it claims to model.** Aerodynamic efficiency of the wing/car as a lift-to-drag-style ratio (printed as "Efficiency (L/D)").

**Literature basis.** The lift-to-drag ratio *is* the standard meaning of "aerodynamic efficiency" (Wikipedia "Lift-to-drag ratio": "It describes the aerodynamic efficiency under given flight conditions" [B11]). In motorsport the same ratio, with downforce as negative lift, is the standard reported metric: Zhang, Toet & Zerihan's race-car ground-effect review frames efficiency exactly as downforce generated per unit drag [B16]; Katz, *Race Car Aerodynamics: Designing for Speed* (1995) [B26] and McBeath, *Competition Car Downforce* (1999) — "the optimum downforce to drag ratio is dependent on the type..." [B31] — use the same metric; contemporary F1 aerodynamic papers report downforce and drag and their ratio (e.g. Basso, Cravero & Marsano, *Energies* 14(8):2059, 2021, on an F1 front wing [B18]). Equivalent coefficient form −C_L/C_D (or |C_z|/C_x) is identical since q·S cancels.

**Assessment: TEXTBOOK_EXACT** (it is the standard metric for ground-effect downforce devices).

**Recommendation.** Two caveats worth one sentence each: (1) report it explicitly as *downforce*-to-drag (−L/D) to avoid sign confusion with aircraft convention; (2) for a front-wing-only surrogate, the ratio is a partial/wing-level efficiency, not the whole-car value, since wheel, body and diffuser drag are excluded or crudely lumped — state that the metric is comparable to published component-level values only with that qualification.

---

## Bibliography

All entries were actually retrieved during this review (pages fetched and read; bibliographic records verified via Crossref/OpenAlex/Open Library).

1. NASA Glenn Research Center, "Induced Drag Coefficient," *Beginner's Guide to Aeronautics*. https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/ (fetched; gives Cdi = Cl²/(π·AR·e), AR = s²/A).
2. NASA Glenn Research Center, "Dynamic Pressure," *Beginner's Guide to Aeronautics*. https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/dynamic-pressure/ (fetched; Bernoulli derivation of q = ρV²/2).
3. NASA Glenn Research Center, "Lift Coefficient," *Beginner's Guide to Aeronautics*. https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/lift-coefficient/ (fetched).
4. Wikipedia, "Prandtl–Glauert transformation." https://en.wikipedia.org/wiki/Prandtl%E2%80%93Glauert_transformation (fetched via Wikipedia API).
5. Wikipedia, "Euler–Bernoulli beam theory" (free-vibration section: ŵₙ modes with βₙ := (μωₙ²/EI)^(1/4); cantilevered-beam example). https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory (fetched).
6. Wikipedia, "Buckling" (plate-buckling derivation: σ_cr = k_cr·π²E/(12(1−ν²)(b/t)²), k_cr = (mb/a + a/mb)², minimum k = 4). https://en.wikipedia.org/wiki/Buckling (fetched).
7. Wikipedia, "Skin friction drag" (Prandtl one-seventh-power law cf = 0.0576·Re_x^(−1/5)). https://en.wikipedia.org/wiki/Skin_friction_drag (fetched).
8. Wikipedia, "von Mises yield criterion." https://en.wikipedia.org/wiki/Von_Mises_yield_criterion (fetched).
9. Wikipedia, "Factor of safety." https://en.wikipedia.org/wiki/Factor_of_safety (fetched).
10. Wikipedia, "Tsai–Wu failure criterion." https://en.wikipedia.org/wiki/Tsai%E2%80%93Wu_failure_criterion (fetched).
11. Wikipedia, "Lift-to-drag ratio." https://en.wikipedia.org/wiki/Lift-to-drag_ratio (fetched).
12. Wikipedia, "Parasitic drag" / "Zero-lift drag coefficient." https://en.wikipedia.org/wiki/Parasitic_drag (fetched).
13. Zerihan, J., and Zhang, X., "Aerodynamics of a Single Element Wing in Ground Effect," *Journal of Aircraft*, Vol. 37, No. 6, pp. 1058–1064, 2000. DOI: 10.2514/2.2711 (record and abstract verified via Crossref/OpenAlex; force-reduction phenomenon below h/c ≈ 0.1 quoted from abstract).
14. Zhang, X., and Zerihan, J., "Aerodynamics of a Double-Element Wing in Ground Effect," *AIAA Journal*, Vol. 41, No. 6, pp. 1007–1016, 2003. DOI: 10.2514/2.2057 (record and abstract verified).
15. Smith, A. M. O., "High-Lift Aerodynamics," *Journal of Aircraft*, Vol. 12, No. 6, pp. 501–530, 1975 (37th Wright Brothers Lecture). DOI: 10.2514/3.59830 (record verified via Crossref).
16. Zhang, X., Toet, W., and Zerihan, J., "Ground Effect Aerodynamics of Race Cars," *Applied Mechanics Reviews*, Vol. 59, No. 1, pp. 33–49, 2006. DOI: 10.1115/1.2110263 (record and abstract verified).
17. Lin, J. C., and Dominik, C. J., "Parametric Investigation of a High-Lift Airfoil at High Reynolds Numbers," *Journal of Aircraft*, Vol. 34, No. 4, pp. 485–491, 1997. DOI: 10.2514/2.2217 (record and abstract verified).
18. Basso, M., Cravero, C., and Marsano, D., "Aerodynamic Effect of the Gurney Flap on the Front Wing of a F1 Car and Flow Interactions with Car Components," *Energies*, Vol. 14, No. 8, 2059, 2021. DOI: 10.3390/en14082059 (record and abstract verified).
19. Kroo, I., "Thin Airfoil Theory," Stanford AA200b Applied Aerodynamics lecture notes (PDF). http://aero-comlab.stanford.edu/aa200b/lect_notes/thinairfoil.pdf (fetched and text-extracted; 2π slope, aerodynamic centre at quarter chord, parabolic-camber example z(x) = 4hx(1−x), and the caveat that inviscid theory overestimates relative to viscous reality).
20. Aviation Stack Exchange: answer a/14559, "What is the method to calculate a finite wing's lift from its sectional airfoil shape?" (2D slope = 2π only for zero thickness; slope increases slightly with thickness; Prandtl–Glauert 1/√(1−M²)) https://aviation.stackexchange.com/a/14559 ; and answer a/111340 (Cl_α = 2π/√(1−M²) derivation) https://aviation.stackexchange.com/a/111340 (both fetched via Stack Exchange API).
21. Blevins, R. D., *Formulas for Natural Frequency and Mode Shape*, Van Nostrand Reinhold, 1979 (bibliographic record verified via Open Library; standard tabulation of β₁L = 1.8751 for cantilevers — the root was additionally verified numerically in this review by solving cosh(x)cos(x) = −1).
22. Schlichting, H., and Gersten, K., *Boundary-Layer Theory* (1st English ed. 1955, Pergamon; McGraw-Hill ed. 1968; bibliographic record verified via Open Library). Cited for turbulent flat-plate skin-friction scaling; the explicit Re_x^(−1/5) law was verified via [B7].
22-T. Timoshenko, S. P., and Gere, J. M., *Theory of Elastic Stability*, 2nd ed., McGraw-Hill, 1961 (bibliographic record verified via Open Library). Cited for the classical plate-buckling stress formula and edge-condition buckling coefficients; the explicit σ_cr formula was verified via [B6].
23. Abbott, I. H., von Doenhoff, A. E., and Stivers, L. S., "Summary of Airfoil Data," NACA Report 824, 1945 (existence as the canonical section-data compendium verified via full-text snippet in [B28]); and Abbott, I. H., and von Doenhoff, A. E., *Theory of Wing Sections*, Dover, 1959 (bibliographic record verified via Open Library). Cited for the realistic magnitude of minimum profile drag (~0.006) of smooth sections.
24. Anderson, J. D., *Fundamentals of Aerodynamics*, 6th ed., McGraw-Hill, 2016/2017 (presence in full-text index and its cambered thin-airfoil-theory coverage verified via Open Library "search inside"). Cited for 2π thin-airfoil slope, cambered-airfoil α_L0 theory (Ch. 4), Prandtl–Glauert (Ch. 9/11), and lifting-line induced drag (Ch. 5).
25. Pope, A., *Basic Wing and Airfoil Theory*, McGraw-Hill, 1951, Ch. 7 "Thin-Airfoil Theory", §2 "The Parabolic-camber Airfoil" (section presence verified via Open Library full-text search). Cited for α_L0 = −2ε for parabolic camber; the result was additionally re-derived analytically in this review.
26. Katz, J., *Race Car Aerodynamics: Designing for Speed*, Robert Bentley, 1995 (bibliographic record and downforce-coefficient content verified via Open Library full-text search).
27. Thomas, F., *Fundamentals of Sailplane Design*, College Park Press, 1999 (full-text snippet verified: "airfoils in inviscid flow have a theoretical lift-curve slope of 2π (6.28) per radian. In practice, viscous [effects reduce it]").
28. Teichmann, F. K., *Airplane Design Manual*, Pitman, 1942 (full-text snippet referencing NACA Report 824 as the summary of airfoil data, verified via Open Library).
29. Jones, R. M., *Mechanics of Composite Materials*, 1975 (Taylor & Francis; bibliographic record verified via Open Library). Cited for orthotropic laminate failure/buckling analysis as the standard replacement for isotropic von Mises / isotropic plate buckling of composite structures.
30. Shigley, J. E., et al., *Shigley's Mechanical Engineering Design*, McGraw-Hill (bibliographic record verified via Open Library). Cited for the factor-of-safety definition and von Mises usage for ductile isotropic materials.
31. McBeath, S., *Competition Car Downforce: A Practical Handbook*, Haynes, 1999 (full-text snippet verified: "the optimum downforce to drag ratio is dependent on the type [of circuit]").

### Honesty notes on the bibliography
- Items [B21], [B22], [B22-T], [B23], [B24], [B25], [B26], [B28], [B29], [B30] are published books whose bibliographic records (and in several cases the relevant section headings or text snippets) were verified via the Internet Archive / Open Library full-text index, but whose full pages were not read cover-to-cover in this review. The specific claims attributed to them are standard textbook content; where a claim rests only on memory of standard content rather than a fetched snippet, it is phrased accordingly above.
- The 0.77 thickness coefficient (formula 1), the 0.5 thickness modifier (formula 2), the 0.02/0.05 drag-buildup coefficients (formula 3), the −0.15 Reynolds exponent (formula 4), the e-degradation per element (formula 5), all ground-effect piecewise coefficients (formulas 6–7), and all slot-effect constants (formula 8) could **not** be found in any published source and are treated as project-specific fits.
