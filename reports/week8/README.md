# PlaylistPulse - Final Project

## Overview

PlaylistPulse is a machine learning project for analyzing Spotify
listening behavior and predicting expected stream count.

## Final Model

**Lasso Regression**

Alpha: `0.0001`

## Final Performance

| Metric | Score |
|---|---:|
| MAE | 0.679296 |
| RMSE | 1.018544 |
| R² | 0.478816 |

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Lasso Regression | 0.679296 | 1.018544 | 0.478816 |
| Gradient Boosting | 0.927447 | 1.115416 | 0.374963 |
| Random Forest | 0.934842 | 1.158535 | 0.325705 |
| Decision Tree | 0.995756 | 1.247293 | 0.218429 |

## Dataset

- Spotify tracks: 114,000 rows
- Spotify history: 149,860 rows
- Matched dataset: 1,710 rows
- Training samples: 1,368
- Testing samples: 342

## Methodology

1. Dataset understanding
2. Exploratory data analysis
3. Data cleaning
4. Feature preprocessing
5. OLS regression
6. Manual Gradient Descent
7. Ridge Regression
8. Lasso Regression
9. Logistic Regression
10. Cross-validation
11. One-standard-error model selection
12. Decision Tree
13. Random Forest
14. Gradient Boosting
15. Final model evaluation
16. SHAP explainability
17. Final integration

## Explainability

SHAP was used to explain the final Lasso model.

Generated explainability outputs include:

- SHAP feature importance
- SHAP bar plot
- SHAP summary plot
- SHAP waterfall plot

## Week 8 Outputs

This folder contains:

- Final model comparison
- Final metrics
- Dataset summary
- Methodology
- Final report
- Presentation summary
- Project summary
- Final model
- Final preprocessor
- Prediction results
- Residual analysis
- Feature importance
- SHAP results
- Final comparison charts

## Final Conclusion

Lasso Regression achieved the strongest final performance among
the evaluated models, with an MAE of 0.679296,
RMSE of 1.018544, and R² of 0.478816.
