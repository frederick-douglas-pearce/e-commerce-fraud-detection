# Cross-Validation Methodology Update

## Problem Identified

The original `bias_variance_analysis.py` script used a **single train/val split** (60/20) to measure train-validation gaps, while GridSearchCV uses **4-fold cross-validation**. This created an inconsistency:

- **GridSearchCV**: Averaged results across 4 folds (more robust)
- **Bias-Variance Script**: Single split results (more prone to random variance)

This led to:
1. Incomparable gap measurements
2. Different optimal n_estimators (95 from GridSearchCV vs 143 from single-split iteration tracking)
3. Less reliable bias-variance diagnostics

## Solution Implemented

Updated `bias_variance_analysis.py` to use **4-fold stratified cross-validation** matching GridSearchCV's methodology:

### Before (Single Split):
```python
# Train on 60% of data
model.fit(X_train, y_train)

# Evaluate on single 20% validation split
train_score = evaluate(X_train, y_train)
val_score = evaluate(X_val, y_val)
gap = train_score - val_score
```

**Issues:**
- Gap depends on random train/val split
- Single validation set can have variance
- Not comparable to GridSearchCV

### After (4-Fold CV):
```python
# Use same CV strategy as GridSearchCV
cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Combine train+val for CV (80% of data, matching GridSearchCV)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Perform 4-fold CV
for fold in cv_strategy.split(X_train_val, y_train_val):
    model.fit(X_fold_train, y_fold_train)
    train_scores.append(evaluate(X_fold_train, y_fold_train))
    val_scores.append(evaluate(X_fold_val, y_fold_val))

# Average across folds
train_score = np.mean(train_scores)
val_score = np.mean(val_scores)
gap = train_score - val_score
```

**Benefits:**
- ✅ Matches GridSearchCV methodology exactly
- ✅ More robust gap estimates (averaged over 4 folds)
- ✅ Comparable to CV results
- ✅ Includes standard deviations showing consistency

## Results Comparison

### Previous (Single Split):
| Model | Train PR-AUC | Val PR-AUC | Gap |
|-------|--------------|------------|-----|
| Logistic Regression | 0.6433 | 0.6530 | -0.0097 |
| Random Forest | 0.9752 | 0.8552 | 0.1200 (12.3%) |
| XGBoost | 0.8788 | 0.8680 | 0.0108 (1.2%) |

### Current (4-Fold CV):
| Model | Train PR-AUC | Val PR-AUC | Gap | Std Dev |
|-------|--------------|------------|-----|---------|
| Logistic Regression | 0.6484 ± 0.0054 | 0.6472 ± 0.0106 | 0.0011 (0.2%) | Low variance ✓ |
| Random Forest | 0.9766 ± 0.0004 | 0.8548 ± 0.0079 | 0.1218 (12.5%) | Very consistent |
| XGBoost | 0.8797 ± 0.0017 | 0.8675 ± 0.0052 | 0.0122 (1.4%) | Low variance ✓ |

### Key Differences:
1. **Standard deviations** now show model consistency across folds
2. **Random Forest** gap slightly higher (12.5% vs 12.3%) but more reliable
3. **XGBoost** gap slightly higher (1.4% vs 1.2%) but averaged over 4 folds
4. **Logistic Regression** shows virtually no gap (0.2%), confirming no overfitting

## Why This Matters

### GridSearchCV Results Are Now Directly Comparable

**GridSearchCV found:** n_estimators=95, PR-AUC=0.8678 (4-fold average)
**Bias-variance script now shows:** PR-AUC=0.8675 (4-fold average)

These match! The script now provides reliable bias-variance diagnostics that align with the hyperparameter tuning methodology.

### Iteration Tracking Discrepancy Explained

**Previous confusion:**
- GridSearchCV: n_estimators=95 optimal
- Single-split iteration tracking: iteration 143 optimal (0.0008 gain)

**Now clear:**
- GridSearchCV uses 4-fold CV (robust)
- Single-split iteration tracking has random variance
- The 0.09% gain at iteration 143 was likely noise
- Gap nearly doubled (1.08% → 2.03%), showing overfitting started

**Recommendation:** Trust GridSearchCV's n_estimators=95

## New Output Files

1. **`01_fold_details.csv`**: Per-fold train/val scores and gaps for each model
   - Shows consistency across folds
   - Identifies if any fold is an outlier

2. **Updated diagnostics**: All gap calculations now use 4-fold averages

## Usage

The script now automatically:
1. Loads best hyperparameters from GridSearchCV results
2. Performs 4-fold CV matching GridSearchCV methodology
3. Averages gaps across folds for robust estimates
4. Provides standard deviations showing model consistency

```bash
uv run python bias_variance_analysis.py
```

Output will show:
```
Train vs Validation Performance (PR-AUC) - 4-FOLD CV AVERAGE
NOTE: These gaps are averaged across 4 folds, matching GridSearchCV methodology

XGBoost:
  Train PR-AUC:      0.8797 ± 0.0017
  Validation PR-AUC: 0.8675 ± 0.0052
  Gap:               0.0122 (1.4%)
  Diagnosis:         ✓ Good fit
```

## Conclusion

The bias-variance analysis script now provides **reliable, GridSearchCV-consistent diagnostics**:

- ✅ Uses same 4-fold CV methodology as hyperparameter tuning
- ✅ Averaged gaps are more robust than single-split estimates
- ✅ Standard deviations show model consistency
- ✅ Directly comparable to GridSearchCV results
- ✅ More trustworthy recommendations

**XGBoost with n_estimators=95 shows excellent generalization (1.4% gap)** and is production-ready.

---

**Updated**: 2025-01-28
**Methodology**: 4-fold stratified cross-validation (matching GridSearchCV)
**Random Seed**: 42 (reproducible)
