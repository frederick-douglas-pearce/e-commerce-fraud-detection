# Notebook Changes Summary

## Changes Made to `fraud_detection_modeling.ipynb`

Based on the bias-variance analysis results, the following changes were implemented to reduce overfitting in Random Forest and XGBoost models.

---

## 1. Random Forest Hyperparameter Grid (Cell 30)

### Previous Configuration (Causing 14.8% Train-Val Gap):
```python
param_grid_rf = {
    'classifier__n_estimators': [400, 450, 500],
    'classifier__max_depth': [25, 30],           # ← Too deep
    'classifier__min_samples_split': [2],         # ← Too aggressive
    'classifier__min_samples_leaf': [2],          # ← Too aggressive
    'classifier__max_features': ['sqrt'],
    'classifier__class_weight': ['balanced_subsample']
}
```

### Updated Configuration (Stronger Regularization):
```python
param_grid_rf = {
    'classifier__n_estimators': [400, 450, 500],
    'classifier__max_depth': [20, 25],           # ✓ REDUCED from [25, 30]
    'classifier__min_samples_split': [5, 10],    # ✓ INCREASED from [2]
    'classifier__min_samples_leaf': [5, 10],     # ✓ INCREASED from [2]
    'classifier__max_features': ['sqrt'],
    'classifier__class_weight': ['balanced_subsample']
}
```

### Expected Impact:
- **Reduce train-val gap** from 14.8% to ~8-10%
- **Training PR-AUC** will decrease from 1.0000 to ~0.92-0.95
- **Validation PR-AUC** should maintain or slightly improve from 0.8523
- **Better generalization** to test set

### Search Space Size:
- Previous: 6 combinations (3 × 2 × 1 × 1)
- Updated: **12 combinations** (3 × 2 × 2 × 2)
- Estimated runtime: ~2-3x longer than previous

---

## 2. XGBoost Hyperparameter Grid (Cell 35)

### Previous Configuration (Causing 12.8% Train-Val Gap):
```python
param_grid_xgb = {
    'classifier__n_estimators': [90, 100, 110],
    'classifier__max_depth': [4, 5],
    'classifier__learning_rate': [0.08, 0.1, 0.12],
    'classifier__subsample': [0.9],
    'classifier__colsample_bytree': [0.9],
    'classifier__min_child_weight': [5],          # ← Insufficient
    'classifier__gamma': [0.5, 0.6],              # ← Insufficient
    # No L1/L2 regularization
    'classifier__scale_pos_weight': [8, 10, 12]
}
```

### Updated Configuration (Stronger Regularization):
```python
param_grid_xgb = {
    'classifier__n_estimators': [85, 90, 95],      # ✓ Centered around optimal 92
    'classifier__max_depth': [4, 5],
    'classifier__learning_rate': [0.08, 0.1, 0.12],
    'classifier__subsample': [0.9],
    'classifier__colsample_bytree': [0.9],
    'classifier__min_child_weight': [7, 10],       # ✓ INCREASED from [5]
    'classifier__gamma': [0.7, 0.8, 1.0],          # ✓ INCREASED from [0.5, 0.6]
    'classifier__reg_alpha': [0.0, 0.1],           # ✓ NEW: L1 regularization
    'classifier__reg_lambda': [1.0, 2.0],          # ✓ NEW: L2 regularization
    'classifier__scale_pos_weight': [8, 10, 12]
}
```

### Expected Impact:
- **Reduce train-val gap** from 12.8% to ~6-8%
- **Training PR-AUC** will decrease from 0.9750 to ~0.90-0.92
- **Validation PR-AUC** should maintain or slightly improve from 0.8497
- **Optimal iteration** already identified at ~92, new grid centers around this

### Search Space Size:
- Previous: 108 combinations (3 × 2 × 3 × 1 × 1 × 1 × 2 × 3)
- Updated: **432 combinations** (3 × 2 × 3 × 1 × 1 × 2 × 3 × 2 × 2 × 3)
- Estimated runtime: ~4x longer than previous
- **Recommendation**: Consider using `search_type='random'` with `n_iter=100` for faster tuning

---

## 3. Final XGBoost Pipeline (Cell 45)

### Previous Configuration:
```python
final_xgb_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=xgb_search.best_params_['classifier__n_estimators'],
        max_depth=xgb_search.best_params_['classifier__max_depth'],
        # ... other params
        # Missing reg_alpha and reg_lambda
        scale_pos_weight=xgb_search.best_params_['classifier__scale_pos_weight'],
        eval_metric='aucpr',
        random_state=random_seed,
        n_jobs=-1
    ))
])
```

### Updated Configuration:
```python
final_xgb_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=xgb_search.best_params_['classifier__n_estimators'],
        max_depth=xgb_search.best_params_['classifier__max_depth'],
        # ... other params
        reg_alpha=xgb_search.best_params_['classifier__reg_alpha'],    # ✓ NEW
        reg_lambda=xgb_search.best_params_['classifier__reg_lambda'],  # ✓ NEW
        scale_pos_weight=xgb_search.best_params_['classifier__scale_pos_weight'],
        eval_metric='aucpr',
        random_state=random_seed,
        n_jobs=-1
    ))
])
```

### Expected Impact:
- Ensures L1/L2 regularization is applied to final production model
- Consistent with hyperparameter tuning configuration

---

## How to Use These Changes

### Option 1: Run Full Hyperparameter Tuning (Recommended for Best Results)

1. **Re-run the Random Forest tuning cell** (Cell 30)
   - Expected runtime: 5-10 minutes (12 combinations × 4 CV folds)
   - Will generate new CV results in `models/logs/random_forest_cv_results_*.csv`

2. **Re-run the XGBoost tuning cell** (Cell 35)
   - Expected runtime: 30-60 minutes (432 combinations × 4 CV folds)
   - **Alternative**: Change `search_type='random'` and `n_iter=100` for ~10 minutes runtime
   - Will generate new CV results in `models/logs/xgboost_cv_results_*.csv`

3. **Re-run remaining cells** to retrain final model and evaluate

### Option 2: Quick Test with Reduced Grid (Faster)

Temporarily reduce the search space for quick testing:

```python
# Quick Random Forest test (4 combinations)
param_grid_rf = {
    'classifier__n_estimators': [450],
    'classifier__max_depth': [20, 25],
    'classifier__min_samples_split': [5],
    'classifier__min_samples_leaf': [5, 10],
    'classifier__max_features': ['sqrt'],
    'classifier__class_weight': ['balanced_subsample']
}

# Quick XGBoost test (24 combinations)
param_grid_xgb = {
    'classifier__n_estimators': [90],
    'classifier__max_depth': [4, 5],
    'classifier__learning_rate': [0.1],
    'classifier__subsample': [0.9],
    'classifier__colsample_bytree': [0.9],
    'classifier__min_child_weight': [7, 10],
    'classifier__gamma': [0.8],
    'classifier__reg_alpha': [0.0, 0.1],
    'classifier__reg_lambda': [1.0, 2.0],
    'classifier__scale_pos_weight': [10]
}
```

### Option 3: Use RandomizedSearchCV (Balanced Approach)

For faster tuning while exploring the full hyperparameter space:

```python
# In both RF and XGBoost tuning cells, change:
search_type = 'random'  # Instead of 'grid'
n_iter = 100  # Test 100 random combinations

# This will:
# - Sample 100 random combinations from the full grid
# - Complete in ~10-15 minutes for XGBoost
# - Provide good coverage of the hyperparameter space
```

---

## Verification Steps

After retraining with new hyperparameters:

1. **Re-run bias-variance analysis**:
   ```bash
   uv run python bias_variance_analysis.py
   ```

2. **Check for improvement**:
   - Train-val gap should be < 10% (down from 12.8-14.8%)
   - Validation PR-AUC should maintain ~0.85 or better
   - Test set PR-AUC should improve or maintain current levels

3. **Review new results**:
   - Check `analysis/bias_variance/01_train_val_gap.png` for updated gaps
   - Review `analysis/bias_variance/04_summary.txt` for new diagnostics

---

## Rollback Instructions

If the new hyperparameters don't improve results, you can revert:

1. **Undo notebook changes**: Use Git to revert the notebook
   ```bash
   git checkout fraud_detection_modeling.ipynb
   ```

2. **Or manually revert** the parameter grids to previous values shown above

---

## Notes

- **Grid size increase**: XGBoost grid increased from 108 to 432 combinations
  - This provides better regularization coverage
  - Consider using RandomizedSearchCV to reduce runtime

- **Early stopping**: Not implemented in final model training (train+val combined)
  - Early stopping requires a validation set to monitor
  - Since final model uses all available data (train+val), no validation set exists
  - The n_estimators from GridSearchCV already reflects optimal stopping

- **Expected gains**: Focus on reducing overfitting, not maximizing validation scores
  - Goal: Better generalization, not higher validation PR-AUC
  - Success = smaller train-val gap + stable/improved test performance

---

**Changes implemented**: 2025-01-XX
**Bias-variance analysis**: `analysis/bias_variance/ANALYSIS_SUMMARY.md`
**Original results**: Saved in `models/logs/` (timestamped files)
