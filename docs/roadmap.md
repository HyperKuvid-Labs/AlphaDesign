# Roadmap

Work is ordered by the evidence needed for a defensible paper:

1. Build an executable equal-budget benchmark for random search, GA-only, and the frozen-policy ablation.
2. Run repeated seeded experiments and report distributions, confidence intervals, feasibility rates, and evaluation counts.
3. Calibrate the empirical surrogate against independent aerodynamic data at matched operating conditions.
4. Validate a small set of representative designs with an external solver or experiment when compute permits.
5. Generate paper tables and figures directly from immutable raw experiment records.

Training a reinforcement-learning policy is out of scope until the pipeline records actions, rewards, and transitions.
