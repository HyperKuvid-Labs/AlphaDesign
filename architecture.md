# AlphaDesign: A Hybrid Actor–Critic Reinforcement Learning and Genetic Algorithm Architecture for Formula 1 Front-Wing Aerodynamic Optimization Under FIA Regulatory Constraints

---

## Abstract

This document describes the architecture and implementation of *AlphaDesign*, an end-to-end optimization system for the parametric design of Formula 1 front wings. The system addresses a high-dimensional, multi-constraint aerodynamic design problem in which an evolving wing geometry must simultaneously maximize downforce and lift-to-drag efficiency, satisfy structural safety requirements, and remain compliant with the Fédération Internationale de l'Automobile (FIA) Formula 1 Technical Regulations. The proposed framework couples three subsystems: (i) a population-based genetic algorithm (GA) operating on a 98-dimensional parametric description of a multi-element wing, (ii) a deep actor–critic reinforcement-learning (RL) agent of the Advantage Actor–Critic (A2C) family that supplies learned, additive parameter refinements during each generation, and (iii) a physics-informed surrogate computational fluid dynamics (CFD) model that performs element-by-element aerodynamic evaluation of the resulting STL geometries. A regulatory layer derived from the FIA 2024 Technical Regulations is embedded directly into the constraint analyzer and into the GA bounds. The system produces, per generation, a checkpointed population, a neural-network state, an STL representation of the best design, and structured JSON aerodynamic reports. This document is the canonical technical reference for the implementation contained in `RL/`; it consolidates information that is otherwise dispersed across module-level scripts and project notes.

---

## 1. Introduction

The design of a Formula 1 front wing is a constrained, multi-objective, high-dimensional engineering problem. The wing must produce on the order of one to two metric tons of downforce at racing speeds while incurring as little drag as possible; it must remain structurally sound under aerodynamic loading, vibration, and impact; it must be manufacturable in carbon-fibre composite within tight tolerances; and it must conform to a regulatory envelope (FIA Technical Regulations, 2024) that fixes the admissible bounding box of the wing and a number of internal geometric features such as the central Y250 region and the endplate proportions. The design space therefore mixes continuous geometric variables, discrete structural choices, and hard inequality constraints.

Two classical approaches are individually inadequate for this problem. A pure genetic algorithm explores the space without bias and tends to generate large numbers of geometrically valid but aerodynamically unpromising candidates, particularly in late generations when the population has converged. A pure deep reinforcement-learning agent, conversely, has difficulty learning a policy directly over a 98-dimensional continuous action space when the reward signal is the output of an expensive aerodynamic simulator and when geometric validity is non-trivial to enforce.

The hypothesis underlying *AlphaDesign* is that the two methods are complementary. A genetic algorithm provides robust, geometry-aware exploration through subsystem-coherent crossover and mutation operators; a learned actor–critic provides a sample-efficient gradient signal that biases each generation's offspring toward higher-fitness regions. The aerodynamic surrogate is positioned between them, fast enough to be invoked on every individual that passes constraint screening, but rich enough to model multi-element interactions, ground effect, and stall behaviour. This document describes the resulting system as it is actually implemented in the `RL/` directory of the repository.

The contributions captured by the implementation are: (1) a 98-dimensional parametric description of a multi-element F1 front wing including endplates, footplates, Y250 region, strakes, pylons, and cascade elements, with bounds taken from FIA Articles 3.3 and 3.4; (2) an A2C-family actor–critic with curriculum learning whose policy outputs are interpreted as bounded additive parameter perturbations applied after the GA's reproductive operators; (3) a physics-informed empirical CFD surrogate that evaluates lift, drag, ground-effect, and slot-gap aerodynamics on a per-element basis from the parameter set and the exported STL; and (4) a fitness function that linearly combines regulatory compliance, an aerodynamic performance index, and the surrogate CFD output, with an adaptive CFD-skip heuristic that allocates compute only to candidates that pass cheap structural and regulatory screens.

---

## 2. Background

The literature touched by this implementation falls into three areas, summarized only to the extent required to situate the design choices.

**Actor–critic reinforcement learning.** The neural component implements an on-policy actor–critic with a shared feature trunk and two heads, one producing a continuous action (a parameter perturbation vector) and the other a scalar value estimate. There is no clipped surrogate objective, no trust region, and no replay buffer; the algorithm therefore lies in the Advantage Actor–Critic (A2C) family rather than in the Proximal Policy Optimization (PPO) family. Curriculum scheduling of the loss weights is layered on top of the standard advantage-weighted policy gradient.

**Genetic algorithms for aerodynamic shape optimization.** The GA operators are not generic; they are parameterized by aerodynamic subsystem (main wing, airfoil, flap stack, endplate, Y250) so that crossover swaps coherent groups of parameters rather than arbitrary scalar fields. Mutation is Gaussian with an adaptive standard deviation that decays linearly with generation index, supplemented by an aggressive variant that drives stronger exploration when needed. Tournament selection at size three is used together with explicit elitism.

**Surrogate aerodynamics.** The CFD subsystem does not solve the Navier–Stokes equations. It is a physics-informed empirical model in which lift and drag coefficients are computed from compressibility-corrected linear airfoil theory with a piecewise stall model, augmented with ground-effect, slot-gap, vortex-induced, and interference terms. The choice is deliberate: a population-scale optimization with hundreds of evaluations per generation cannot afford a high-fidelity CFD solve per individual, and a surrogate that is differentiable in spirit and tunable to F1-specific phenomena (Y250 vortex, ground effect under skirts, multi-element circulation boost) is more useful for rapid iteration than a generic external solver.

---

## 3. Problem Formulation

### 3.1 Design parameter space

The optimization variable is an instance of the dataclass `F1FrontWingParams` declared in `RL/formula_constraints.py`. It contains 98 fields organized into 13 categories. Scalar fields cover main-wing planform, airfoil thickness and camber distribution, endplate proportions, footplate geometry, Y250 region, pylon mounting, cascade elements, manufacturing wall thicknesses, mesh-generation hyperparameters, and material and target performance values. Several fields are array-valued and parameterize the per-flap properties of a three-flap stack.

The thirteen categories are summarized below; the complete listing including units and bounds is provided in Appendix A.

```
Category                       Example fields                                       Count
-----------------------------------------------------------------------------------------
Main wing structure            total_span, root_chord, tip_chord, sweep_angle         5
Airfoil profile                max_thickness_ratio, camber_ratio, camber_position     7
Flap system (3 flaps)          flap_spans, flap_root_chords, flap_slot_gaps, ...     24
Endplate system                endplate_height, endplate_max_width, endplate_lean     7
Y250 vortex region             y250_step_height, y250_transition_length               4
Footplate                      footplate_extension, arch_radius                       4
Strake system                  primary_strake_count, strake_heights                   2
Pylon mounting                 pylon_count, pylon_spacing, pylon_major_axis           5
Cascade elements               primary_cascade_span, secondary_cascade_chord          5
Manufacturing                  wall_thickness_*, minimum_radius                       4
Mesh / construction            resolution_span, resolution_chord, mesh_density        6
Material and targets           density, target_downforce, efficiency_factor           5
Compliance / limits            (clip ranges; not optimised)                          20
-----------------------------------------------------------------------------------------
Total                                                                                98
```

The lower and upper bounds applied during initialization and by the post-mutation clipper are partially regulatory and partially engineering. Wing-span bounds, for example, are taken from FIA 2024, Article 3.3.1; root- and tip-chord bounds from Article 3.3.2; endplate height and maximum width from Articles 3.4.1 and 3.4.2; Y250 step height from Article 3.3.6.

### 3.2 Optimization statement

Let `θ ∈ Θ ⊂ ℝ^{98}` denote the parameter vector and `Θ` the admissible set defined by the FIA-derived bounds. The system maximizes a scalar fitness `F(θ)` defined as a convex combination of three components:

```
F(θ) = 0.30 · F_constraint(θ) + 0.40 · F_performance(θ) + 0.30 · F_CFD(θ) + B(θ)
```

where `F_constraint` is a regulatory and structural compliance score, `F_performance` is an analytical aerodynamic-quality score, `F_CFD` is the score returned by the surrogate CFD, and `B(θ)` is a small bonus term granted to designs that simultaneously exceed thresholds on all three components and on the static safety factor (see Section 9). All three components are normalized to `[0, 100]`.

The optimization is therefore a single-objective scalarization of an underlying multi-objective problem. This was chosen, rather than maintaining a Pareto front, because the GA selection operators and the actor–critic value head both consume a scalar fitness signal and because the regulatory and structural floors function as soft penalties through `F_constraint`.

### 3.3 Constraints

The admissible set `Θ` is defined by:

* **Regulatory constraints.** FIA-derived box bounds on every continuous parameter. These are enforced by clipping after every variation operator.
* **Structural constraints.** Static safety factor (Von Mises stress relative to ultimate tensile strength), buckling safety factor, and natural frequency. These are computed in `F1FrontWingAnalyzer` and enter the fitness through `F_constraint` and the CFD-skip heuristic.
* **Manufacturing constraints.** Minimum radius and minimum wall thicknesses that distinguish structural, aerodynamic, and detail regions.
* **Subsystem coherence.** Implicit constraints maintained by the GA mutation operator: `chord_taper_ratio = tip_chord / root_chord`, monotonically non-increasing flap spans, `endplate_min_width < endplate_max_width`.

---

## 4. System Architecture

### 4.1 Top-level data flow

```
                                config.json                base_params.json
                                    |                            |
                                    v                            v
                              +-----------------------------------+
                              |    alphadesign.py  (entry point)  |
                              +------------------+----------------+
                                                 |
                                                 v
                              +-----------------------------------+
                              |       AlphaDesignPipeline         |
                              |       (main_pipeline.py)          |
                              +-+--------------+----+-------------+
                                |              |    |
        initialize_pipeline_components         |    |  finalize_pipeline
                                |              |    |
                                v              v    v
                  +------------------+   +-----------------+   +-----------------+
                  | Genetic Algo     |   | Actor-Critic NN |   | Constraint      |
                  | components       |   | components      |   | analyzer        |
                  | (genetic_algo_*) |   | (neural_network_|   | (formula_       |
                  |                  |   |   components)   |   |  constraints.py)|
                  +-------+----------+   +--------+--------+   +--------+--------+
                          |                       |                     |
                          +-----------+-----------+---------------------+
                                      |
                                      v
                          +-----------------------+
                          | UltraRealisticF1      |
                          | FrontWingGenerator    |
                          | (wing_generator.py +  |
                          |  generation_scripts/) |
                          +-----------+-----------+
                                      |
                                      v   STL + cfd_params.json
                          +-----------------------+
                          | STLWingAnalyzer       |
                          | (cfd_analysis.py)     |
                          +-----------+-----------+
                                      |
                                      v
                          +-----------------------+
                          | Output artifacts      |
                          | checkpoints/          |
                          | stl_outputs/          |
                          | cfd_results/          |
                          | neural_networks/      |
                          | logs/                 |
                          +-----------------------+
```

### 4.2 Pipeline orchestrator

`AlphaDesignPipeline` (`RL/main_pipeline.py`, lines 24–1034) is the central object. Its lifetime is structured into three phases:

1. **Initialization** — `initialize_pipeline_components` (lines 147–171) instantiates the GA components (population initializer, crossover operator, mutation operator), the fitness evaluator, and, when enabled in `config.json`, the actor–critic network.
2. **Optimization loop** — `run_optimization_loop` (lines 203–257) repeatedly invokes `run_single_generation` (lines 259–386), checkpoints the state at a configurable cadence, and breaks out when an early-stopping criterion is satisfied.
3. **Finalization** — `finalize_pipeline` (lines 998–1011) writes the final neural-network weights and a JSON summary report.

### 4.3 Module map

```
File / Directory                                      Responsibility
------------------------------------------------------------------------------
RL/alphadesign.py                                     CLI entry point, base-parameter loading
RL/main_pipeline.py                                   AlphaDesignPipeline orchestrator
RL/formula_constraints.py                             F1FrontWingParams, F1FrontWingAnalyzer
RL/wing_generator.py                                  UltraRealisticF1FrontWingGenerator (geometry)
RL/generation_scripts/                                Specialized geometry generators
RL/cfd_analysis.py                                    STLWingAnalyzer (surrogate CFD)
RL/run_cfd_with_json.py                               JSON-driven CFD entry point
RL/neural_network_components/                         Actor-critic network, loss, optimizer, action mapping
RL/genetic_algo_components/                           GA: init, crossover, mutation, fitness
RL/early_stopping.py                                  EarlyStoppingManager
RL/config.json                                        Pipeline hyperparameters
RL/checkpoints/, stl_outputs/, cfd_results/, logs/    Runtime artifacts
```

---

## 5. Geometry Generation Subsystem

### 5.1 Parametric wing synthesis

The class `UltraRealisticF1FrontWingGenerator` in `RL/wing_generator.py` (lines 26–2243) consumes an `F1FrontWingParams` instance and produces a single multi-element STL mesh together with a JSON file that records the geometric parameters used by the CFD subsystem. The generator is parametric down to per-flap angle of attack, slot-gap profile, gurney-flap height, endplate forward lean, and Y250 step transition.

Mesh-generation resolution is controlled by `resolution_span`, `resolution_chord`, and `mesh_density`. A typical wing produces of the order of 10⁵ vertices and 2×10⁵ faces. A `surface_smoothing` pass (eight iterations by default) is applied prior to STL export.

### 5.2 Airfoil construction

Each individual element is built from a cubic Bézier representation of the upper and lower surface, using cosine-spaced parametric points to densify the leading-edge region:

```
P(t)  = (1 - t)^3 · P_0 + 3(1 - t)^2 t · P_1 + 3(1 - t) t^2 · P_2 + t^3 · P_3
t     = 0.5 · (1 - cos(π · t_raw)),   t_raw ∈ [0, 1]
```

Control-point ordinates are modulated by the camber and thickness fields of `F1FrontWingParams`. The base profile is a modified NACA 64A010, retaining the 6-series laminar-flow heritage that is well suited to ground-effect operation; thickness distribution follows the standard four-digit form

```
y_t(x) = 0.5 · t · (0.2969·sqrt(x) - 0.126·x - 0.3516·x^2 + 0.2843·x^3 - 0.1015·x^4)
```

with `t` the maximum thickness ratio.

### 5.3 Multi-element layout

The default configuration is a four-element stack: a main element plus three flaps. Each flap has an independent root and tip chord, camber, slot-gap, vertical and horizontal offsets relative to the upstream element, and a per-flap geometric angle of attack. The slot-gap profile is not a constant offset but a longitudinally varying corridor: the gap factor opens at the entry, narrows through the throat, and reopens at the exit, with an entry angle of approximately 7.5° and an exit angle of approximately 3.5° at the default settings (`wing_generator.py`, lines 748–797).

### 5.4 Specialized generators

Three modular generators in `RL/generation_scripts/` are integrated by the master generator:

* **`f1_main_wing_geometry.py` — `F1FrontWingMainElementGenerator`** implements the main-element camber line including a Stratford-criterion separation factor (`β = 0.39`), a peak suction coefficient `C_p,min = -5.8` placed near 30% chord, an explicit Venturi factor for the underside, and a parameterized ride-height sensitivity over the operational range 30–80 mm.
* **`f1_multi_flap_system_gen.py` — `F1FrontWingMultiElementGenerator`** applies progressive angle-of-attack to each flap with default slot-gap-to-chord ratios of `[1.2%, 1.0%, 0.8%]`, gurney-flap heights of `[3, 2.5, 2] mm`, and per-flap leading-edge droop and trailing-edge kick-up controls used as anti-stall features.
* **`f1_y250_gen.py` — `F1FrontWingY250CentralStructureGenerator`** generates the regulated 500 mm central section, including primary strakes (count limited to two by FIA post-2019 rules), outboard fences, and elliptical mounting pylons with default major and minor axes of 38 mm and 25 mm.

### 5.5 STL export

Geometry is exported through `trimesh` to a single binary STL file in `RL/stl_outputs/` (per-generation best designs) or in `RL/f1_wing_output/` (specialized generator outputs). A companion file with suffix `_cfd_params.json` records the exact geometric quantities required by the surrogate CFD, so that downstream analysis is not forced to re-derive them from the mesh; this avoids small but consequential discrepancies between the quantity that drives the geometry and the quantity that drives the analysis.

---

## 6. Surrogate CFD Subsystem

### 6.1 Model class

The CFD subsystem is implemented in `RL/cfd_analysis.py` (≈2,200 lines), centred on the class `STLWingAnalyzer`. It is a physics-informed empirical surrogate, not a Navier–Stokes solver. The model is composed of: a per-element compressibility-corrected lift law with a piecewise stall extension; a profile-plus-induced drag law with element-dependent Oswald efficiency; a piecewise ground-effect amplifier on the main element only; a slot-gap interaction model that boosts circulation on downstream elements as a function of gap-to-chord ratio and overlap; and an environmental correction layer that adjusts air density and Mach number for ambient conditions.

The model loads geometry preferentially from the JSON sidecar (`load_geometry_from_json`, `cfd_analysis.py`, lines 167–205) and falls back to mesh-derived auto-detection if the JSON is absent. The principal entry point is `multi_element_analysis` (lines 711–878), and a higher-level sweep harness `run_comprehensive_f1_analysis` (lines 963–1079) iterates speed, angle, ride-height, and environmental ranges to produce sensitivity reports.

### 6.2 Lift model

`enhanced_airfoil_lift_coefficient` (lines 519–570) implements:

```
β        = sqrt(1 - M^2)                            (Prandtl-Glauert factor)
Cl_alpha = (2π / β) · (1 + 0.77 · t/c)               (compressibility & thickness)
α_0      = -2 · m · (1 + 0.5 · t/c)                  (zero-lift angle from camber)
Cl_0     = Cl_alpha · α_0
Cl_lin   = Cl_alpha · α + Cl_0                       (linear region)

if α > α_stall:
    Cl   = Cl_stall · exp(-(α - α_stall) / k_decay)  (post-stall decay)
    Cl   = max(Cl, 0.3 · sin(α))                      (post-stall floor)
```

with element-dependent stall angle (≈18° for the main element, ≈22° for flaps) reflecting the higher stall margin available to highly cambered, slot-fed downstream elements.

### 6.3 Drag model

`enhanced_airfoil_drag_coefficient` (lines 572–632) sums profile and induced contributions with explicit Reynolds and Mach corrections:

```
Cd_0      = 0.006 + 0.02 · m + 0.05 · (t/c)^2        (profile / zero-lift drag)
re_factor = (Re / 1e6)^(-0.15)        if Re > 1e6
mach_fac  = 1 + 5 · (M - 0.3)^2       if M > 0.3
e_oswald  = 0.7 - 0.05 · element_idx
Cd_i      = Cl^2 / (π · AR_element · e_oswald)        (induced drag)
Cd        = (Cd_0 · re_factor · mach_fac) + Cd_i
```

Element-by-element decrement of Oswald efficiency captures the finite-aspect-ratio penalty incurred by downstream flaps relative to the main element.

### 6.4 Ground-effect model

`calculate_ground_effect` (lines 634–662) is applied only to the main element (`element_idx = 0`) and is piecewise in the height-to-chord ratio `h/c`:

```
For h/c < 0.10 :   F_ground = 2.20 - 1.20 · (h/c)
For 0.10 ≤ h/c < 0.50 : F_ground = 1.00 + 1.20 · exp(-3 · h/c)
For h/c ≥ 0.50 : F_ground = 1.00 + 0.20 · exp(-h/c)
```

Downstream elements receive an exponentially attenuated share of the ground-effect amplification, scaled by `0.8^element_idx`, reflecting that their effective ground clearance is determined by the main element's wake rather than by the floor itself.

### 6.5 Multi-element slot-gap model

`calculate_slot_effect` (lines 664–709) consumes the JSON-recorded gap and overlap ratios and produces multiplicative coefficients that act on the lift and drag of the downstream element:

```
gap_efficiency   = exp(-((gap_ratio - 0.02) / 0.01)^2)
Γ_boost          = 1.30 + 0.15 · gap_efficiency · overlap_efficiency
velocity_ratio   = 1.40 + 0.40 · gap_efficiency
slot_Cl_mult     = Γ_boost · sqrt(velocity_ratio)
slot_Cd_mult     = 0.85 + 0.15 · gap_efficiency
```

The gap-efficiency Gaussian is centred at `gap_ratio = 0.02`, consistent with the empirical observation that a gap of approximately 2% of chord yields the strongest circulation enhancement before slot-flow detachment dominates.

### 6.6 Multi-element analysis loop

`multi_element_analysis` performs:

1. Per-element Reynolds number `Re = ρ V c / μ`.
2. Per-element environmental correction of air density and speed of sound.
3. Per-element call to the lift and drag models.
4. Per-element application of the ground-effect amplifier and the slot-gap multiplier inherited from the upstream element.
5. Force integration on the per-element reference area at dynamic pressure `q = 0.5 · ρ · V^2`.
6. Aggregate KPIs: total downforce, total drag, lift-to-drag, downforce-to-weight ratio (assuming a 1500 kg car), centre-of-pressure location, balance coefficient, stall margin, and yaw sensitivity.

---

## 7. Genetic Algorithm Subsystem

### 7.1 Population initialization

`F1PopulInit` (`RL/genetic_algo_components/initialize_population.py`, lines 7–115) constructs the initial population by inserting the base design as the first individual and generating `population_size − 1` variants via `create_f1_variant`. Each scalar parameter is perturbed multiplicatively with a per-category variation factor (for example `0.10` for `total_span`, `0.15` for `root_chord`, `0.20` for `max_thickness_ratio`, `0.25` for `camber_ratio` and `sweep_angle`), and the result is clipped to the regulatory bounds. Array parameters such as `flap_cambers` and `flap_slot_gaps` are perturbed element by element. The variation rule is:

```
θ_new = clip( θ_base · (1 + variation_factor · (U(0,1) - 0.5) · 2),  θ_min, θ_max )
```

Parameter bounds are stored in a dictionary (lines 49–81) and reflect the FIA articles cited in Section 3.

### 7.2 Selection

Selection is by tournament. The tournament size is three by default (`AlphaDesignPipeline.tournament_selection`, `main_pipeline.py`, lines 446–451). The top 20% of each generation is preserved as elite (lines 421–430) and copied unchanged into the next generation, which provides a lower bound on best-fitness regression.

### 7.3 Crossover

Crossover is implemented in `F1AeroCrossover.f1_aero_crossover` (`RL/genetic_algo_components/crossover_ops.py`, lines 20–53). Rather than swapping arbitrary scalar fields, it swaps coherent aerodynamic subsystems with the following independent probabilities:

```
Subsystem               Swap probability    Fields swapped
-------------------------------------------------------------------------------------
Main wing               0.40                total_span, root_chord, tip_chord, sweep
Airfoil profile         0.30                max_thickness_ratio, camber_ratio, position
Flap system             0.50                flap_spans, flap_root_chords, flap_tip_chords,
                                            flap_cambers, flap_slot_gaps, flap_offsets
Endplate system         0.30                endplate_height, endplate_max_width,
                                            endplate_lean, endplate_sweep, wrap
```

The overall recombination decision is gated by a `crossover_rate` (default 0.8). Subsystem-level swapping preserves the engineering relationships between fields that must vary together.

### 7.4 Mutation

`F1MutationOperator.f1_wing_mutation` (`RL/genetic_algo_components/mutation_strategy.py`, lines 21–109) applies multiplicative Gaussian mutation per parameter:

```
θ_new = θ · (1 + N(0, σ_mutation))
```

with `σ_mutation = 0.5` by default. Adaptive scheduling (`adaptive_f1_mutation`, lines 111–121) anneals σ linearly with generation index:

```
σ_mutation(g) = σ_0 · (1 - g / G_max)
```

so that exploration dominates early generations and is gradually replaced by exploitation. An `aggressive_mutation` variant (lines 222–241) raises mutation rate and strength temporarily and applies targeted perturbations to flap-system, weight-related, and structural parameters; this is intended for stagnation recovery.

After mutation, `_ensure_system_coherence` (lines 202–220) restores subsystem invariants:

```
chord_taper_ratio              ← tip_chord / root_chord
flap_spans[i]                  ≤  flap_spans[i-1] · 0.95     (monotone non-increasing)
endplate_min_width             <  endplate_max_width
```

### 7.5 Hyperparameter table

```
Hyperparameter             Default value         Source
-----------------------------------------------------------------------------
population_size            20–50                 config.json
elitism fraction           0.20                  main_pipeline.py
tournament size            3                     main_pipeline.py
crossover_rate             0.80                  crossover_ops.py
mutation_rate              0.60                  mutation_strategy.py
mutation σ_0               0.50                  mutation_strategy.py
σ schedule                 σ_0 · (1 - g / G_max) mutation_strategy.py
max_generations            50 (default)          config.json
```

---

## 8. Reinforcement-Learning Subsystem

### 8.1 Network architecture

The actor–critic network is defined in `RL/neural_network_components/forward_pass.py`. It consists of a shared feature trunk that maps the flattened parameter vector into a hidden representation, followed by two heads.

```
Input  : θ ∈ ℝ^{n}        (n ≈ 60-70 after dataclass flattening)
                                 |
Trunk:  three blocks of  Linear(d_in → d_hidden)
                          LayerNorm(d_hidden)
                          GELU
                          Dropout(0.10)
        d_hidden = min(512, max(4 · n, 256))
                                 |
        +---------------------+--+--------------------+
        |                                              |
   Policy head                                    Value head
   (policy_head.py)                              (value_head.py)
   Linear(d → 2d) → LN → GELU → Dropout(0.10)    Linear(d → d/2) → LN → GELU → Dropout(0.10)
   Linear(2d → d) → LN → GELU → Dropout(0.05)    Linear(d/2 → d/4) → LN → GELU → Dropout(0.05)
   Linear(d → n) → Tanh                           Linear(d/4 → 1)
```

Weights are initialized through `NetworkInitializer.setup_network` (`network_initialization.py`, lines 5–56) using Kaiming-normal initialization on the linear layers (mode `fan_out`, nonlinearity `relu`) and unit/zero initialization on `LayerNorm` parameters. Device placement is automatic.

### 8.2 Policy head

The policy head is an expand-then-compress feedforward block whose output is bounded into `[−1, 1]` by a `Tanh` activation. The output dimensionality matches the parameter count, so each scalar of the policy can be interpreted as a normalized perturbation direction along the corresponding axis of the parameter space.

### 8.3 Value head

The value head is a contracting feedforward block producing a single scalar. The value is interpreted as an estimate of the design's normalized fitness and is the target of a Huber regression loss against the surrogate CFD score (Section 8.5).

### 8.4 Action mapping (RL → GA bridge)

`ParamterTweaker.apply_neural_tweaks` (`RL/neural_network_components/parameter_tweaking.py`, lines 10–32) converts the policy output into a parameter perturbation:

```
Δθ      = tanh(π(s)) · mod_scale         with mod_scale = 0.10
ε       ~ N(0, 0.01^2)                   exploration noise
θ_new   = clip(θ + Δθ + ε,  θ_min, θ_max)
```

The policy is therefore not used as a generator of complete designs; it is used as a residual refinement applied on top of the GA-generated population at each generation. The tanh bound, the small `mod_scale`, and the per-axis clipping together ensure that no single neural step can leave the regulatory envelope or violate the FIA bounds.

The conversion between the dataclass population and the flat tensor is handled by `genetic_to_neural_params` (lines 34–104) and its inverse `tensor_to_individual` (defined in `main_pipeline.py`).

### 8.5 Loss function

The composite loss is implemented in `AlphaDesignLoss` (`RL/neural_network_components/loss_calculation.py`, lines 5–65). With `π` denoting the policy output, `V̂` the value head output, `s_CFD` the surrogate CFD score for the corresponding individual, and `Δ` the parameter improvement actually realized by the action, the components are:

```
L_value  = HuberLoss(V̂, s_CFD)
L_policy = - E[ π · tanh(Δ / 10) · σ(s_CFD / 50) ]            (advantage-weighted)
L_reg    = E[ π^2 ] + 0.1 · E[ (π_{i+1} - π_i)^2 ]            (L2 + smoothness)
L_total  = w_v · L_value + w_π · L_policy + 0.01 · L_reg
```

with default weights `w_v = w_π = 1.0`. The advantage term is `σ(s_CFD / 50)`, mapping the (positive-going) CFD score into a `(0, 1)` factor, and the improvement term `tanh(Δ / 10)` saturates extreme parameter swings so they do not dominate the policy gradient.

The form is consistent with on-policy advantage actor–critic; there is no clipped surrogate (PPO), no replay buffer (SAC), and no multi-step return target. The smoothness penalty `(π_{i+1} - π_i)^2` discourages high-frequency oscillation across consecutive individuals in a batch, which empirically reduces sudden population drift between adjacent generations.

### 8.6 Optimizer

`OptimizerManager` (`RL/neural_network_components/optimizer_integration.py`, lines 5–82) defaults to AdamW (`lr = 1e-3`, `weight_decay = 1e-4`, `betas = (0.9, 0.999)`), with an enhanced configuration `use_adamw_cosine` that pairs AdamW with `CosineAnnealingWarmRestarts`:

```
T_0   = 10        (initial restart period, in epochs)
T_mult = 2        (period doubles after each restart: 10, 20, 40, ...)
eta_min = 1e-6
```

The warm-restart schedule is intended to escape local minima in the policy landscape, which is known to be non-convex in this regime.

### 8.7 Curriculum learning

`train_neural_network_extended` (`main_pipeline.py`, lines 725–828) trains the actor–critic against the recent generation history with three weight regimes:

```
Phase    Generations  w_constraint  w_performance  Epochs
-------------------------------------------------------------
Early    g <  40       0.6           0.4            40
Middle   40 ≤ g < 80   0.4           0.6            25
Late     g ≥  80       0.2           0.8            15
```

Constraint-loss and performance-loss are each MSE losses against, respectively, the `F1FrontWingAnalyzer` compliance score and the normalized fitness. Gradient clipping (`clip_grad_norm_` at norm 1.0) is applied before the optimizer step. Training terminates early within an epoch budget if the average loss falls below `1e-6`.

### 8.8 Algorithm classification

The combination of a separate value head used as a baseline, advantage-weighted policy gradient, on-policy data, and absence of trust-region or clipped surrogates places the implementation in the A2C family rather than in PPO or DDPG/SAC. The curriculum schedule is an additional layer that does not change the underlying class.

---

## 9. Fitness Evaluation and Reward Shaping

### 9.1 Constraint fitness

`FitnessEval.evaluate_formula_constratins` (`RL/genetic_algo_components/fitness_evaluation.py`, lines 195–298) calls `F1FrontWingAnalyzer.run_complete_analysis` to compute a base compliance percentage and adds bonuses for satisfaction of higher-order desiderata (optimal flap gap, multi-element effectiveness, downforce/efficiency targets, adequate safety, regulatory compliance, Y250 compliance):

```
base_compliance         ∈ [0, 1]   (overall_compliance from F1FrontWingAnalyzer)
compliance_bonus        =  0.08 · I[flap_gap_optimal]
                        +  0.06 · I[multi_element_effective]
                        +  0.12 · I[downforce_target_met]
                        +  0.10 · I[efficiency_target_met]
                        +  0.10 · I[safety_factor_adequate]
                        +  0.06 · I[buckling_safe]
                        +  0.05 · I[span_regulation_compliant]
                        +  0.05 · I[y250_compliance]
adjusted_compliance     =  min(1.0, base_compliance + compliance_bonus)
F_constraint            =  100 · adjusted_compliance
```

### 9.2 Performance fitness

`F_performance` (lines 522–601) combines aerodynamic and structural sub-scores with the analytically computed downforce and lift-to-drag ratio:

```
F_performance = min(100,
        35 · (downforce / 1200)
      + 25 · efficiency
      + 20 · (safety_factor / 4.0)
      + 10 · (structural_score / 100)
      + 10 · (aerodynamic_score / 100) )
```

The per-component sub-scores (`structural_score`, `aerodynamic_score`) are themselves computed in `_compute_structural_score` (lines 300–345) and `_compute_aerodynamic_score` (lines 347–394) by piecewise mapping of safety factor, natural frequency, buckling safety, mass, efficiency, Reynolds number, ground-effect factor, and multi-element effectiveness onto bonuses out of 100.

### 9.3 CFD fitness

When the CFD-skip heuristic does not skip a candidate:

```
F_CFD = min(100,
        50 · (cfd_downforce / 1500)
      + 30 · cfd_efficiency
      + 20 · (stall_margin / 10) )  +  Q
```

with `Q ∈ {-10, 0, 5, 10, 15}` a discrete CFD-quality bonus driven by the analyzer's qualitative rating (`Excellent`, `Good`, `Acceptable`, `Poor`).

### 9.4 Total fitness and exceptional-design bonus

```
F = 0.30 · F_constraint + 0.40 · F_performance + 0.30 · F_CFD

if  F_constraint > 90  AND  F_performance > 80  AND  F_CFD > 75  AND  safety_factor ≥ 3.0 :
    F += 10
```

### 9.5 Adaptive CFD-skip heuristic

`FitnessEval.should_skip_cfd` (lines 34–47) bypasses the surrogate CFD whenever any of the following holds:

```
adjusted_compliance         < 0.40
safety_factor               < 1.50
buckling_safety_factor      < 1.50
natural_frequency           < 15 Hz
```

Skipped individuals receive a conservative fitness derived from constraint and performance components only. This heuristic, implemented in `evaluate_cfd_perf` (lines 396–444) as a guard, removes between 40% and 60% of CFD calls in typical runs without measurable effect on the elite individuals, since failing candidates would not be selected regardless.

---

## 10. FIA Regulatory Compliance Layer

The compliance layer is constituted by (i) parameter bounds applied during initialization and after every variation operator, and (ii) more than thirty validation predicates evaluated by `F1FrontWingAnalyzer.run_complete_analysis` (`RL/formula_constraints.py`, lines 977–1046). The article-by-article correspondence with FIA 2024 Section C is:

```
Parameter / predicate                FIA Article           Bound (default range)
----------------------------------------------------------------------------------
total_span                           Article 3.3.1         [1600, 1800] mm
root_chord, tip_chord                Article 3.3.2         [200, 350] / [250, 330] mm
y250_step_height                     Article 3.3.6         [15, 50] mm
y250_width (fixed)                   Article 3.3.6         500 mm
endplate_height                      Article 3.4.1         [250, 325] mm
endplate_max_width                   Article 3.4.2         [80, 150] mm
flap_count                           Section C, multi-elem ≤ 5 (default 3)
slot_gaps                            Multi-element rules   [8, 20] mm
ground_clearance (operational)       Floor regulations     ≥ 75 mm
primary_strake_count                 Post-2019 limits      ≤ 2
```

The analyzer produces a dictionary of boolean validations consumed by the constraint-fitness computation, plus auxiliary computed values (Reynolds numbers, ground-effect factor, mass, natural frequency, buckling safety) reused by the surrogate aerodynamic and structural sub-scores.

The constraint analyzer also encodes a body of analytical aerodynamic and structural physics that is too large to reproduce here; representative formulas, verbatim from the source, include the Y250 vortex circulation `Γ_Y250 = (h_step/1000) · V / (L_trans/1000)^2`, the induced angle of attack `α_i = atan(Γ_tip / (V · b/4))`, the maximum bending moment for a simply-supported wing `M_max = w · b^2 / 8`, the Von Mises combined stress `σ_vm = sqrt(σ_bend^2 + 3 · τ^2)`, and the first-mode natural frequency `f_n = (1.875^2 / 2π) · sqrt(E·I / (μ · L^4))`.

---

## 11. Training and Optimization Loop

The per-generation algorithm executed by `run_single_generation` (`RL/main_pipeline.py`, lines 259–386) is:

```
for g = 0, 1, ..., G_max - 1:

    # 1. Fitness evaluation
    for each θ_i in current_population:
        c_i  = evaluate_formula_constraints(θ_i)
        if should_skip_cfd(c_i):
            cfd_i = default_conservative_cfd()
        else:
            STL_i = UltraRealisticF1FrontWingGenerator(θ_i).export()
            cfd_i = STLWingAnalyzer(STL_i).multi_element_analysis()
        f_i  = combine_scores(c_i, cfd_i)

    # 2. Selection and reproduction
    elite        = top_20_percent_by_fitness(current_population, f)
    new_pop      = elite.copy()
    while |new_pop| < population_size:
        p_a, p_b = tournament_select(current_population, k=3),
                   tournament_select(current_population, k=3)
        c_a, c_b = f1_aero_crossover(p_a, p_b)
        c_a, c_b = f1_wing_mutation(adaptive_sigma(g))(c_a),
                   f1_wing_mutation(adaptive_sigma(g))(c_b)
        new_pop.extend([c_a, c_b])

    # 3. Neural-network guidance (if enabled)
    if config.neural_network_enabled:
        for θ_i in new_pop:
            t_i              = individual_to_tensor(θ_i)
            (π_i, V̂_i)       = network(t_i)
            t_i              = apply_neural_tweaks(t_i, π_i, exploration=True)
            θ_i              = tensor_to_individual(t_i, θ_i)

    # 4. Network training (curriculum-weighted)
    if g mod training_frequency == 0:
        epochs           = curriculum_epochs(g)
        w_c, w_p         = curriculum_weights(g)
        for e in 1..epochs:
            for h in recent_generation_buffer:
                t            = genetic_to_neural_params(h.best_individual)
                π, V̂         = network(t)
                L_c          = MSE(V̂, h.compliance)
                L_p          = MSE(V̂, normalize(h.fitness))
                L            = w_c · L_c + w_p · L_p
                optimizer.zero_grad()
                L.backward()
                clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

    # 5. Checkpointing and best-design persistence
    if g mod save_frequency == 0:
        save_checkpoint(g, new_pop, network)
        save_best_design_stl(argmax_f(new_pop), g)

    # 6. Early stopping (currently optional)
    if early_stopping_manager.should_stop(history_of_best_fitness):
        break

    current_population = new_pop
```

`EarlyStoppingManager` in `RL/early_stopping.py` (lines 4–65) supports three criteria simultaneously: a patience criterion (no improvement greater than `min_delta = 0.005` for `patience = 30` generations), a stagnation-detection criterion on the last `stagnation_threshold` scores, and a population-diversity convergence threshold. `config.json` declares the parameters (`patience`, `min_delta`, `monitor`, `stagnation_threshold`, `convergence_threshold`).

The checkpoint format written by `_save_checkpoint` (lines 646–721) is:

```json
{
  "generation": 10,
  "timestamp": "ISO8601",
  "config": { ... },
  "current_population": [ { ... }, ... ],
  "best_designs_history": [ { "gen": 0, "fitness": ..., "individual": { ... } }, ... ],
  "generation_results": [ { "gen": 0, "best_fitness": ..., "avg_fitness": ..., ... }, ... ],
  "neural_network_checkpoint": "neural_networks/network_gen_010.pth"
}
```

---

## 12. Implementation Details

### 12.1 Languages and libraries

The implementation is written in Python (≥3.8). The principal third-party dependencies are PyTorch (actor–critic network, optimizers, LR schedulers), NumPy and SciPy (numerical computation throughout the surrogate CFD and the constraint analyzer), and `trimesh` (mesh construction, smoothing, STL export, mesh-based geometry queries). A `requirements.txt` and a leaner `requirements_minimal.txt` are provided.

### 12.2 Output artifact tree

```
RL/
├── checkpoints/
│   ├── checkpoint_gen_000.json
│   ├── summary_gen_000.json
│   └── final_summary.json
├── stl_outputs/
│   └── generation_<g>_best_design.stl
├── neural_networks/
│   ├── network_gen_<g>.pth
│   ├── training_metrics_gen_<g>.json
│   └── final_network.pth
├── cfd_results/
│   └── gen<g>_ind<i>_cfd_results.json
├── f1_wing_output/
│   └── *.stl  +  *_cfd_params.json    (specialized-generator outputs)
└── logs/
    └── alphadesign_<YYYYMMDD>_<HHMMSS>.log
```

The per-individual CFD result file records the input parameters, the integrated forces, the per-element breakdown, and the F1-specific KPIs, so that any individual can be re-analyzed offline without re-running the GA.

### 12.3 JSON-driven CFD integration

`RL/run_cfd_with_json.py` (`run_cfd_with_accurate_parameters`) is the standalone entry point used to evaluate a previously generated wing. It instantiates `STLWingAnalyzer` with both the STL path and the parameter JSON, performs the full sweep through `run_comprehensive_f1_analysis`, and emits a comparison report contrasting the JSON-declared geometry with the geometry auto-detected from the mesh; this is used to detect drift between geometry specification and geometry production. The JSON schema is documented in `RL/CFD_JSON_INTEGRATION.md`.

### 12.4 Unit conventions

The internal unit conventions are: linear dimensions in millimeters, angles in degrees (converted to radians at the analytical boundary), velocities in metres per second (converted from km/h at the user boundary), forces in newtons, masses in kilograms, frequencies in hertz, pressures in pascals, and densities in kg/m³. All conversions happen at module boundaries so that the constraint analyzer and CFD module can assume SI inside their numerics.

---

## 13. Discussion

### 13.1 Strengths

The system's principal strengths are physics-informed evaluation, sample efficiency, and regulation-awareness.

The surrogate CFD is detailed enough to model the qualitative phenomena that distinguish a Formula 1 front wing from a generic multi-element wing: ground effect on the main element, slot-gap circulation boost, Y250 vortex circulation, endplate-driven outwash, and stall margin under combined yaw and ride-height variation. This provides a fitness landscape with structure relevant to the engineering objective, rather than a smooth surrogate fit that can be exploited by adversarial geometries.

The hybrid operator is sample-efficient relative to a pure RL baseline because the GA already produces feasible designs, and the actor–critic only needs to learn a small residual perturbation. The action space is small in magnitude (`mod_scale = 0.10`), which limits the variance of the policy gradient and makes the value-head regression target stable enough for Huber regression.

Regulatory compliance is enforced at three layers (initialization clipping, post-mutation clipping, and constraint-fitness penalty), which makes it difficult for the optimizer to discover an out-of-envelope design even by chance.

### 13.2 Limitations

Several limitations are inherent to the implementation. First, the CFD is a surrogate, not a Navier–Stokes solver; it captures qualitative aerodynamic behaviour but not the full three-dimensional flow field, and its quantitative outputs should be calibrated against high-fidelity simulation or wind-tunnel data before being used for downstream decisions. Second, the RL component is on-policy and uses a small recent-generation buffer rather than a true replay buffer, which limits sample reuse; an off-policy method (e.g., SAC) over the same parameterization could be more sample-efficient but would require explicit handling of the bounded continuous action space. Third, the optimization currently terminates on a per-run generation budget (default 50), with early stopping available but optional in the present codebase. Fourth, the fitness scalarization implies a fixed trade-off between downforce, drag, and structural margin; if a Pareto front is required, an explicit multi-objective evolutionary algorithm such as NSGA-II would be a more direct fit.

### 13.3 Alignment with the project roadmap

The repository's root `README.md` lists the closing of the main pipeline integration, the maturation of the neural-network training loop, GPU acceleration, hyperparameter tuning, and a multi-objective extension as the open work items. The architecture described above explicitly accommodates each of these: the pipeline integration is captured by `AlphaDesignPipeline`, the training loop by `train_neural_network_extended`, GPU usage is automatic through PyTorch device handling, hyperparameters are externalized in `config.json`, and the fitness scalarization could be replaced by a Pareto-rank computation without modifying the GA operators or the actor–critic.

---

## 14. Conclusion

*AlphaDesign* combines a subsystem-coherent genetic algorithm, an A2C-family actor–critic, and a physics-informed empirical CFD surrogate into a single optimization pipeline for FIA-compliant Formula 1 front wings. The architecture exploits the complementary strengths of the three subsystems: the GA provides geometry-aware exploration, the actor–critic provides learned residual refinement, and the surrogate provides per-element aerodynamic feedback at a cost low enough for population-scale evaluation. The 98-dimensional parameterization, the FIA-derived bounds, and the multi-component fitness function tie the optimization to the engineering and regulatory context. The implementation in `RL/` realizes this architecture, with explicit checkpointing, curriculum learning, adaptive CFD skipping, and full traceability of every generated design to its STL geometry and JSON aerodynamic report.

---

## References

The references cited inline in this document are intentionally informal; this is an internal architectural document rather than a peer-reviewed submission. The following are the primary sources actually used by the implementation or by the analytical formulations.

* FIA, *2024 Formula 1 Technical Regulations*, Issue 8, October 2024 (`fia_2024_formula_1_technical_regulations_-_issue_8_-_2024-10-17.pdf`, repository root).
* FIA, *2026 Formula 1 Regulations — Section C — Technical*, Issue 13, July 2025 (`fia_2026_f1_regulations_-_section_c_technical_-_iss_13_-_2025-07-31.pdf`, repository root).
* Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning* (2016) — A2C/A3C.
* Abbott and von Doenhoff, *Theory of Wing Sections* — NACA airfoil families and analytical aerodynamic coefficients.
* Stratford, *The Prediction of Separation of the Turbulent Boundary Layer* (1959) — separation criterion used in the main-element generator.
* Katz and Plotkin, *Low-Speed Aerodynamics* — multi-element wing theory and induced-drag formulation.

---

## Appendix A — Parameter Schema (`F1FrontWingParams`)

```
Field                             Unit          Default        Bound (where applicable)
------------------------------------------------------------------------------------------
total_span                        mm            1600           [1600, 1800]    (FIA 3.3.1)
root_chord                        mm             280           [250,  330]    (FIA 3.3.2)
tip_chord                         mm             250           [200,  330]    (FIA 3.3.2)
chord_taper_ratio                 -              0.89          derived
sweep_angle                       deg            3.5           [0,    8]
dihedral_angle                    deg            2.5           [0,    6]
twist_distribution_range          deg           [-1.5, 0.5]    -

base_profile                      str            NACA_64A010_modified
max_thickness_ratio               -              0.15          [0.04, 0.20]
camber_ratio                      -              0.08          [0.06, 0.15]
camber_position                   -              0.40          [0.35, 0.50]
leading_edge_radius               mm             2.8           -
trailing_edge_thickness           mm             2.5           -
upper_surface_radius              mm             800           -
lower_surface_radius              mm            1100           -

flap_count                        -              3             ≤ 5
flap_spans                        mm            [1600,1500,1400]    -
flap_root_chords                  mm            [220,180,140]       -
flap_tip_chords                   mm            [200,160,120]       -
flap_cambers                      -             [0.12,0.10,0.08]    -
flap_slot_gaps                    mm            [14,12,10]          [8, 20]
flap_vertical_offsets             mm            [25,45,70]          -
flap_horizontal_offsets           mm            [30,60,85]          -

endplate_height                   mm             280           [250, 325]    (FIA 3.4.1)
endplate_max_width                mm             120           [80,  150]    (FIA 3.4.2)
endplate_min_width                mm              40           min < max
endplate_thickness_base           mm              10           -
endplate_forward_lean             deg              6           -
endplate_rearward_sweep           deg             10           -
endplate_outboard_wrap            deg             18           -

footplate_extension               mm              70           -
footplate_height                  mm              30           -
arch_radius                       mm             130           -
footplate_thickness               mm               5           -

primary_strake_count              -                2           ≤ 2 (post-2019)
strake_heights                    mm            [45, 35]       -

y250_width                        mm             500           fixed (FIA 3.3.6)
y250_step_height                  mm              18           [15, 50]   (FIA 3.3.6)
y250_transition_length            mm              80           [80, 120]
central_slot_width                mm              30           [0,  30]

pylon_count                       -                2           -
pylon_spacing                     mm             320           -
pylon_major_axis                  mm              38           -
pylon_minor_axis                  mm              25           -
pylon_length                      mm             120           -

cascade_enabled                   bool          True           -
primary_cascade_span              mm             250           -
primary_cascade_chord             mm              55           -
secondary_cascade_span            mm             160           -
secondary_cascade_chord           mm              40           -

wall_thickness_structural         mm               4.0         -
wall_thickness_aerodynamic        mm               2.5         -
wall_thickness_details            mm               2.0         -
minimum_radius                    mm               0.4         -

mesh_resolution_aero              -                0.4         -
mesh_resolution_structural        -                0.6         -
resolution_span                   -               40           -
resolution_chord                  -               25           -
mesh_density                      -                1.5         -
surface_smoothing                 bool          True           -

material                          str            Standard Carbon Fiber
density                           kg/m^3        1600           -
weight_estimate                   kg               4.0         -

target_downforce                  N             4000           -
target_drag                       N               40           -
efficiency_factor                 -                1.0         -
```

---

## Appendix B — File-to-Concept Map

```
File or directory                                         Section(s)
----------------------------------------------------------------------------
RL/alphadesign.py                                          §4.1, §4.2
RL/main_pipeline.py                                        §4, §8.7, §11
RL/config.json                                             §4.1, §11
RL/formula_constraints.py                                  §3.1, §3.3, §10
RL/wing_generator.py                                       §5
RL/generation_scripts/f1_main_wing_geometry.py             §5.4
RL/generation_scripts/f1_multi_flap_system_gen.py          §5.4
RL/generation_scripts/f1_y250_gen.py                       §5.4, §10
RL/cfd_analysis.py                                         §6
RL/run_cfd_with_json.py                                    §12.3
RL/CFD_JSON_INTEGRATION.md                                 §12.3
RL/neural_network_components/forward_pass.py               §8.1
RL/neural_network_components/policy_head.py                §8.2
RL/neural_network_components/value_head.py                 §8.3
RL/neural_network_components/network_initialization.py     §8.1
RL/neural_network_components/parameter_tweaking.py         §8.4
RL/neural_network_components/loss_calculation.py           §8.5
RL/neural_network_components/optimizer_integration.py      §8.6
RL/genetic_algo_components/initialize_population.py        §7.1
RL/genetic_algo_components/crossover_ops.py                §7.3
RL/genetic_algo_components/mutation_strategy.py            §7.4
RL/genetic_algo_components/fitness_evaluation.py           §9
RL/early_stopping.py                                       §11
RL/SYSTEM_FLOWCHARTS.md                                    §4.1
RL/SPECIALIZED_GENERATORS_INTEGRATION.md                   §5.4
RL/FIA_COMPLIANCE_UPDATE_SUMMARY.md                        §10
fia_2024_formula_1_technical_regulations_*.pdf (root)      §3, §10, References
fia_2026_f1_regulations_*.pdf (root)                       §3, §10, References
```
