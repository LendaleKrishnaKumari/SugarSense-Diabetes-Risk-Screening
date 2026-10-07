# Bonus 3 — Mini App

A simple patient diabetes-risk screening function was created using the best tuned machine-learning model.

## Risk Categories

| Predicted Probability | Risk Level |
|---|---|
| < 0.30 | Low |
| 0.30 – < 0.60 | Medium |
| ≥ 0.60 | High |

## Model

The selected best tuned model was saved using Joblib:

`sugarsense_model.joblib`

## Test Result

The Mini App was tested successfully.

Example output:

```text
Patient Risk: Medium
