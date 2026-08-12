# RISED Framework Specification

RISED is a structured pre-deployment evaluation approach spanning five dimensions.
Each dimension is operationalized through formally specified, measurable sub-criteria
grounded in the published evaluation, fairness, and calibration literature. The
formal definition of each metric is in the module docstring of its
implementation; this page is a map to those definitions, not a duplicate of
them.

## Dimension 1: Reliability

Judge Sensitivity Score and per-perturbation rank correlation under
semantics-preserving input perturbations. Implementation and formal
definition: `rised/reliability.py`, `rised/perturbations.py`.

## Dimension 2: Inclusivity

Maximum per-partition AUC parity gap across demographic and clinical
subgroups. Implementation and formal definition: `rised/inclusivity.py`.

## Dimension 3: Sensitivity

Maximum threshold flip rate over a band of decision thresholds.
Implementation and formal definition: `rised/sensitivity.py`.

## Dimension 4: Equity

Spearman correlation between predicted risk and an independent need proxy,
reported against its attainable ceiling. Implementation and formal
definition: `rised/equity.py`.

## Dimension 5: Deployability

Single-row and batch scoring latency, explanation stability. Implementation
and formal definition: `rised/deployability.py`.

## References

See `../Paper_healthAI_decision_sup/references.bib` for the full bibliography.
