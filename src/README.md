# src/ Folder Structure

This folder contains all source code for the E-Commerce Fraud Detection project, organized into deployment and notebook-specific utilities.

## Folder Organization

```
src/
├── deployment/          # Production code for model deployment
│   ├── config/          # Configuration management
│   ├── data/            # Data loading and splitting
│   ├── evaluation/      # Model evaluation and metrics
│   └── preprocessing/   # Feature engineering pipeline
├── fd1_nb/             # Utilities for notebook #1 (EDA & Feature Engineering)
├── fd2_nb/             # Utilities for notebook #2 (TBD)
└── fd3_nb/             # Utilities for notebook #3 (TBD)
```

## Subfolders

### deployment/
**Purpose:** Production code for model training, evaluation, and API deployment.

**Modules:**
- **config/** - Configuration management
  - `data_config.py` - Data paths, split ratios, random seeds
  - `model_config.py` - Model hyperparameters, feature lists
  - `training_config.py` - Cross-validation strategy, threshold targets

- **data/** - Data loading and splitting utilities
  - `loader.py` - `load_and_split_data()` function for consistent train/val/test splitting

- **evaluation/** - Model evaluation and threshold optimization
  - `metrics.py` - `calculate_metrics()`, `evaluate_model()` functions
  - `thresholds.py` - `optimize_thresholds()` for precision/recall trade-offs

- **preprocessing/** - Production feature engineering pipeline
  - `config.py` - `FeatureConfig` dataclass for deployment configuration
  - `features.py` - Feature engineering functions (timezone conversion, temporal features, etc.)
  - `pipelines.py` - `PreprocessingPipelineFactory` for sklearn preprocessing
  - `transformer.py` - `FraudFeatureTransformer` (sklearn-compatible transformer)

**Used by:** `train.py`, `predict.py`, `bias_variance_analysis.py`

### fd1_nb/
**Purpose:** General-purpose utility functions for exploratory data analysis and feature engineering in the first notebook.

**Modules:**
- `data_utils.py` - Data loading, splitting, target analysis, feature statistics
- `eda_utils.py` - EDA functions (VIF, correlations, distributions, temporal analysis, mutual information)
- `feature_engineering.py` - General-purpose feature engineering utilities (timezone conversion, temporal features, interaction features, percentile-based features)

**Used by:** Notebook #1 (EDA & Feature Engineering)

**Design Philosophy:** General-purpose, configurable functions that work with any dataset when given proper parameters. Includes verbose output and visualizations for exploratory analysis.

### fd2_nb/
**Purpose:** Utility functions for the second notebook (TBD).

**Status:** Placeholder for future notebook utilities.

### fd3_nb/
**Purpose:** Utility functions for the third notebook (TBD).

**Status:** Placeholder for future notebook utilities.

## Import Patterns

### From Deployment Scripts

```python
# train.py, predict.py, bias_variance_analysis.py
from src.deployment.config import DataConfig, ModelConfig, FeatureListsConfig, TrainingConfig
from src.deployment.data import load_and_split_data
from src.deployment.evaluation import evaluate_model, optimize_thresholds
from src.deployment.preprocessing import FraudFeatureTransformer, PreprocessingPipelineFactory
from src.deployment.preprocessing.config import FeatureConfig
```

### From Notebooks

```python
# Notebook #1 (EDA & Feature Engineering)
from src.fd1_nb.data_utils import (
    load_data, split_train_val_test, analyze_target_stats, analyze_feature_stats
)
from src.fd1_nb.eda_utils import (
    analyze_vif, analyze_correlations, analyze_mutual_information,
    plot_numeric_distributions, plot_box_plots, analyze_temporal_patterns
)
from src.fd1_nb.feature_engineering import (
    convert_utc_to_local_time, create_temporal_features,
    create_interaction_features, create_percentile_based_features
)
```

## Docker Deployment

To deploy only production code, copy only the `deployment/` folder:

```dockerfile
# Dockerfile
COPY src/deployment/ /app/src/deployment/
```

This ensures minimal image size and excludes notebook-specific utilities from production.

## Design Decisions

### Deployment vs Notebook Code

**Deployment Code (`deployment/`):**
- Production-ready, optimized for performance
- Fraud detection-specific (hardcoded thresholds, domain logic)
- Minimal dependencies, strict error handling
- Designed for consistent, reproducible model training and inference

**Notebook Code (`fd1_nb/`, `fd2_nb/`, `fd3_nb/`):**
- Exploratory, general-purpose utilities
- Configurable via parameters (no hardcoded values)
- Verbose output, visualizations for analysis
- Designed for reusability across different datasets and projects

### Redundancy Trade-off

Some code overlap exists between deployment and notebook modules (e.g., timezone conversion, temporal feature extraction). This is intentional:
- **Deployment code** is optimized for production use cases (specific thresholds, error handling)
- **Notebook code** is flexible for exploration (configurable parameters, visualization)
- Minimal shared complexity keeps Docker deployment simple (no shared dependencies)

Future optimization: If redundancy becomes problematic, consider extracting truly shared utilities into a `src/shared/` module.

## File Count

- **Deployment modules:** 12 Python files across 4 subfolders
- **Notebook modules:** 3 Python files (fd1_nb), with fd2_nb and fd3_nb placeholders
- **Total:** 15+ Python files, ~3,000+ lines of code

## Testing

Unit tests are organized to mirror the source structure:
- `tests/test_eda/` - Tests for `src/fd1_nb/` modules (45 tests)
- Future: `tests/test_deployment/` - Tests for `src/deployment/` modules

Run tests:
```bash
pytest tests/                    # All tests
pytest tests/test_eda/          # Notebook utility tests only
```
