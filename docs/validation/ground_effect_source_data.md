# Ground-effect downforce vs ride height — real, citable data findings

Compiled 2026-09-17. Everything below was fetched and read this session. Primary artifacts saved:
- `/tmp/zerihan_thesis.pdf` — J. Zerihan, PhD thesis, Univ. of Southampton (2001), 236 pp
  (source: eprints.soton.ac.uk/426058/1/00192634.pdf via Wayback Machine snapshot 2026-05-12)
- `/tmp/zhang_toet_zerihan_2006_review.pdf` — Zhang, Toet & Zerihan, "Ground Effect Aerodynamics
  of Race Cars", Applied Mechanics Reviews 59(1), 2006, DOI 10.1115/1.2110263
  (source: eprints.soton.ac.uk/42969/1/GetPDFServlet.pdf via Wayback snapshot 2026-02-16)
- `/tmp/zhang_zerihan_2003_double_element.pdf` — Zhang & Zerihan, "Aerodynamics of a
  Double-Element Wing in Ground Effect", AIAA Journal 41(6), 2003, DOI 10.2514/2.2057
  (source: eprints.soton.ac.uk/22605/1/AIAA-2057-653.pdf via Wayback snapshot 2026-04-13)

Common experiment context (thesis Ch.2; AIAA 2003 paper):
- Wing section: Tyrrell 026 F1 front-wing profile, 80% scale; span 1100 mm, endplates fitted.
- Single element: chord c = 223.4 mm, AR 4.92, Re ≈ 0.43–0.46e6 (30 m/s, moving ground).
- Double element: total chord c = 380 mm (flap chord 165.7 mm), Re ≈ 0.735–0.765e6, moving ground.
- Incidence convention: quoted α measured on a reference line; true α = quoted + 2.45°.
  "Reference incidence" α = 1° quoted = 3.45° true (this is why the review paper says α = 3.45°).
- Force measurement uncertainty (AIAA 2003 paper, p.1009): CL ±0.003, CD ±0.0006.

================================================================================
1. SINGLE-ELEMENT WING — exact numbers stated in text (read directly, high confidence)
================================================================================

Zerihan thesis, Ch.3/4/5/6 (transition-free unless noted), α = 1° quoted (3.45° true):

- Freestream (h/c = 3.36):            CL = 0.69   (thesis §4.2)
- h/c = 0.671:                        CL = 0.84   (thesis §4.2)
- Freestream CLmax (vs incidence):    CL = 1.35 at α = 11.3° quoted (thesis §3.2; lift slope 4.57/rad)
- PEAK in ground effect:              CL = 1.72 at h/c = 0.082  (thesis §4.2 and §4.7)
- Slope dCL/d(h/c) between h/c = 0.179 and 0.134 (force-enhancement region, thesis §5.2.1):
    4.3 at α=1°, 2.9 at α=3°, 1.8 at α=5°, 1.1 at α=7° (quoted incidences)
- CL gain from h/c = 0.671 to peak (thesis §5.2.1):
    +0.90 (α=1°), +0.82 (α=3°), +0.75 (α=5°), +0.70 (α=7°)
- Highest CL anywhere in the study:   CLmax = 2.26 at h/c = 0.134, α = 12.3° quoted (thesis §5.2.2)
- Fixed transition (thesis §6.3, review §4.5):
    peak CL = 1.39 at h/c = 0.112; freestream CLmax = 1.22 at α = 9° (§6.2);
    below the peak the curve is "plateau shaped" — much milder drop than transition-free.
- Gurney flaps (thesis §7.3): small Gurney peak CL = 2.27 at h/c = 0.112;
    large Gurney peak CL = 2.52 at h/c = 0.112 (clean wing: 1.72 at 0.082).

Zhang, Toet & Zerihan (2006) review, §4.4–4.6 (same experiments; read directly from text):
- "Fixed transition reduces CLMAX from 1.72 to 1.39."
- "The corresponding increases in downforce from freestream to the respective maximum are
   141% for the free transition case and 117% for the fixed transition case."
   (i.e. peak/freestream = 2.41x free-transition, 2.17x fixed-transition; implies freestream
   CL ≈ 0.71 — thesis says 0.69, which gives 2.49x. Both are citable; the small mismatch
   is the authors' own rounding.)
- "The height at which maximum downforce occurs increases from h = 0.08c for the free
   transition case to h = 0.112c for fixing transition."
- On Ranzenbach & Barlow (2D NACA 4412, Re = 1.5e6, stationary ground): "downforce was seen
   to reach a maximum at a height of approximately 0.08c for a single element aerofoil" (review §4.4).

================================================================================
2. SINGLE-ELEMENT WING — digitized from figures (visual estimates, ±~0.01 CL, ±~0.005 h/c)
================================================================================

Thesis Figure 7a (p.46 of thesis; PDF p.64) — CL vs h/c, α = 1° quoted, transition free.
Same data as review Fig. 8a (which adds the fixed-transition curve and extends to h/c = 1.0):

  h/c     CL (transition free)     CL (transition fixed, review Fig. 8a only)
  0.055   ~1.60 (leftmost point)   ~1.35
  0.060   ~1.65                    ~1.36
  0.067   ~1.68                    ~1.37
  0.075   ~1.71                    ~1.38
  0.082   ~1.72  <-- PEAK          ~1.38
  0.090   ~1.72                    ~1.39  <-- fixed-transition PEAK (at 0.112 per text)
  0.110   ~1.71                    ~1.39
  0.134   ~1.61                    ~1.34
  0.179   ~1.41                    ~1.19
  0.224   ~1.26                    (not separately read)
  0.313   ~1.08                    ~1.04
  0.448   ~0.93                    ~0.89
  0.671   ~0.84 (text: 0.84)       ~0.82
  1.000   ~0.77 (review Fig. 8a)   ~0.77
  freestream (h/c=3.36): 0.69 (text value)

Confidence: visually estimated from rendered 200-dpi page images; anchored by the exact text
values (peak 1.72 @ 0.082, CL(0.671)=0.84, freestream 0.69), which the figure reproduces.

================================================================================
3. DOUBLE-ELEMENT WING — exact numbers (Zhang & Zerihan 2003 AIAA J.; thesis Ch.8)
================================================================================

Main element at reference incidence, flap at optimum overlap 0.024c / gap 0.032c:

- Low flap angle (datum − 8.5°):
    PEAK: CL = 2.588 at h/c = 0.066  (AIAA paper §"Results", thesis §8.2 — read from text)
    slope discontinuity (plateau) at h/c = 0.171
- High flap angle (datum):
    PEAK: CL = 3.028 at h/c = 0.079  (AIAA paper, thesis §8.2 — read from text)
    discontinuity at h/c = 0.237, "downforce actually reduces suddenly just below this height"
- +9.5° flap deflection: lower downforce at practically all heights; flap stalled even at
    large height (thesis §8.2).

Digitized from AIAA paper Fig. 5a (visual estimates):

  h/c     CL (low flap)    CL (high flap)
  0.050   ~2.44            ~2.47
  0.055   ~2.52            ~2.58
  0.060   ~2.57            ~2.65
  0.066   ~2.588 PEAK      ~2.63   (sharp drop zone below peak for high flap)
  0.073   ~2.57            ~2.98
  0.079   ~2.55            ~3.028 PEAK
  0.10    ~2.48            ~2.96
  0.13    ~2.35            ~2.87
  0.171   ~2.24 (plateau)  ~2.65
  0.21    ~1.96            ~2.64 (slight local uptick)
  0.237   ~1.85            ~2.53 (a/b boundary)
  0.30    ~1.65            ~2.28
  0.395   ~1.51            ~2.08

Freestream CL for the double-element wing is NOT stated numerically in any fetched text
(freestream tests at h/c = 1.97 were done but the value appears only in figures that do not
cover that height). Honest bound from Fig. 5a trend: both curves are still falling steeply at
h/c = 0.395, so freestream CL is below ~1.5 (low flap) / ~2.1 (high flap), making the peak
gain at least ~1.7x–1.5x respectively — do not quote a precise freestream value for the
double-element case.

================================================================================
4. PEAK LOCATION AND MAGNITUDE SUMMARY
================================================================================

Single element (Tyrrell 026 profile, α=3.45° true, Re≈4.5e5, moving ground, endplates):
- Peak at h/c = 0.082 (free transition), CL = 1.72 = 2.41–2.49x freestream (0.69–0.71).
- Peak at h/c = 0.112 (fixed transition), CL = 1.39 = 2.17x freestream.
- 2D NACA 4412 (Ranzenbach & Barlow, Re=1.5e6, stationary ground; quoted in thesis §4.7 and
  review §4.4 — secondary citation, primary paper NOT fetched):
    experiment: peak CL ≈ 0.93 at h/c ≈ 0.08–0.09 vs far-field CL ≈ 0.41 (≈2.27x);
    their moving-ground CFD: peak CL ≈ 1.06.

Double element (Re≈7.5e5, moving ground):
- Low flap: peak h/c = 0.066, CL = 2.588.
- High flap: peak h/c = 0.079, CL = 3.028.

So for both configurations the peak sits at h/c ≈ 0.066–0.112, NOT at h/c → 0.

================================================================================
5. SEPARATION ONSET AND DOWNFORCE FALL BELOW THE PEAK
================================================================================

Single element, transition free (thesis §4.3.1):
- Trailing-edge separation first appears (tiny region, x_sep ≈ 0.99c) at h/c = 0.224 —
  i.e. separation onset is well ABOVE the downforce peak; the peak (0.082) occurs when the
  separated region has grown large ("large scale separation, i.e. stall", review §4.4).
  A constant-pressure (separated) region near the TE is visible for h/c < 0.179 and grows
  as h/c reduces.
- Fall below peak: CL 1.72 @ h/c=0.082 → ~1.60 @ h/c≈0.055 (lowest measured),
  ≈ −7% over Δh/c ≈ 0.027 (slope ≈ 4.4 CL per unit h/c). Note the enhancement-side slope
  just above the peak is of similar magnitude (4.3 per unit h/c between 0.179 and 0.134).
- Fixed transition: fall below the 1.39 peak is much milder — plateau-like, ~1.35 at
  h/c ≈ 0.05 (≈ −3%).

Double element (AIAA 2003):
- High flap: "sharp reduction just beyond the maximum, mainly because of the boundary layer
  separating, and a resultant loss of circulation on the main element." CL 3.028 @ 0.079 →
  ~2.63 @ 0.066 → ~2.47 @ 0.05: ≈ −18% over Δh/c ≈ 0.03 (very steep, ~3–4x steeper than
  the single-element case).
- Low flap: CL 2.588 @ 0.066 → ~2.44 @ 0.05: ≈ −6%, more gradual.

================================================================================
6. WHAT FAILED / NOT OBTAINED
================================================================================

- websearch skill unusable (no Serper API key configured) — all searching done via
  DuckDuckGo HTML scraping, Semantic Scholar / OpenAlex / Unpaywall APIs, and Wayback CDX.
- eprints.soton.ac.uk is behind the Anubis anti-bot proof-of-work (HTTP 401/403 to plain
  HTTP clients). All Southampton PDFs were obtained via Wayback Machine snapshots instead.
- Zerihan & Zhang 2000 J. Aircraft single-element paper (DOI 10.2514/2.2711): no open copy
  exists (closed on Semantic Scholar, OpenAlex, Unpaywall). The old AIAA preview URL
  (pdf.aiaa.org/jaPreview/...) is dead and not archived. HOWEVER: the thesis (same author,
  same experiments, Ch.4–6) and the 2006 review Fig. 8 reproduce the identical data, so the
  numbers above are the paper's underlying dataset from the primary author.
- EThOS record for the thesis exists (uk.bl.ethos.392649) but downloads need BL login;
  not needed since the eprints PDF was recovered.
- Ranzenbach & Barlow papers (AIAA 95-1909 / 96-2469 class, and AIAA 97-2238 multi-element)
  and Jasinski & Selig SAE 983042: paywalled, not fetched. Numbers quoted above from these
  are secondary citations via the thesis/review and are marked as such.
- Katz "Race Car Aerodynamics" / McBeath "Competition Car Downforce": books not freely
  available; not consulted.
- Semantic Scholar rate-limited (HTTP 429) on the last two auxiliary searches; not retried
  further since the core dataset was already complete.

================================================================================
7. IMPLICATION FOR THE SURROGATE (calculate_ground_effect)
================================================================================

Current code rises monotonically to 2.2–2.5 as h/c → 0. The real data show:
- multiplier vs freestream peaks at ≈ 2.4–2.5x (single element, free transition) at
  h/c ≈ 0.08 — so the current cap of 2.5 is accidentally about right AT h/c ≈ 0.08 only;
- below h/c ≈ 0.08 the multiplier must FALL (to ≈ 2.3x at h/c ≈ 0.055 for the single
  element; much steeper, to well under 2x by h/c ≈ 0.05, for a high-flap-angle two-element
  wing);
- at h/c = 0.671 the multiplier is only ~1.22x (0.84/0.69), and ~1.1x at h/c = 1.0
  (0.77/0.71) — the current code's mild-decay branch (1.0 + 0.2·e^-h/c) overestimates the
  mid-range and the invented 2.2-at-zero branch is wrong in sign of the trend.
