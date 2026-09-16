# Legacy artifacts

`artifacts/legacy/` contains generated meshes, logs, checkpoints, model
weights, and other historical outputs moved out of the active source tree.
They are retained for provenance only. They are invalid legacy artifacts, not
paper evidence or reproducible experiment results.

New experiments should write to an explicitly recorded run directory and keep
the seed, configuration, code version, and evaluation budget alongside any
results. The current `experiment_runner` records metadata only; it does not
claim to have executed the listed strategies.
