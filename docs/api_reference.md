# API Reference

<!-- Populated in Session 3 once implementations are complete. -->

## Top-level entry point

- `rised.evaluate_all(model, X, y_true, demographic_df, ...)` → `FrameworkReport`

## Dimension modules

- `rised.reliability.evaluate_reliability(...)`
- `rised.inclusivity.evaluate_inclusivity(...)`
- `rised.sensitivity.evaluate_sensitivity(...)`
- `rised.equity.evaluate_equity(...)`
- `rised.deployability.evaluate_deployability(...)`

## Result dataclasses

- `ReliabilityResult`
- `InclusivityResult`
- `SensitivityResult`
- `EquityResult`
- `DeployabilityResult`
- `FrameworkReport`

## Utilities

- `rised.metrics` — core metric functions
- `rised.perturbations` — perturbation generators
- `rised.datasets` — Synthea cohort loading
- `rised.visualization` — plot functions
