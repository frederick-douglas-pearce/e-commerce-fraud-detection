"""
Model configuration for fraud detection project.

Centralizes model hyperparameters and feature lists loading to ensure
consistency across all scripts (bias_variance_analysis.py, train.py).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class FeatureListsConfig:
    """Configuration for feature categorization."""

    DEFAULT_FEATURE_LISTS_PATH = Path("models/feature_lists.json")
    DEFAULT_METADATA_PATH = Path("models/model_metadata.json")

    @classmethod
    def load(cls, source: str = "feature_lists.json") -> Dict[str, List[str]]:
        """Load feature categorization from configuration file.

        Args:
            source: Either "feature_lists.json" or "model_metadata.json"

        Returns:
            Dictionary with keys: 'continuous_numeric', 'categorical', 'binary', 'all_features'

        Raises:
            FileNotFoundError: If the specified configuration file doesn't exist
        """
        if source == "feature_lists.json":
            path = cls.DEFAULT_FEATURE_LISTS_PATH
        elif source == "model_metadata.json":
            path = cls.DEFAULT_METADATA_PATH
        else:
            path = Path(source)

        if not path.exists():
            raise FileNotFoundError(
                f"Feature configuration file not found: {path}\n"
                f"Expected either {cls.DEFAULT_FEATURE_LISTS_PATH} or {cls.DEFAULT_METADATA_PATH}"
            )

        with open(path, 'r') as f:
            data = json.load(f)

        # Handle both formats
        if source == "model_metadata.json":
            # Extract features section from model metadata
            if 'features' in data:
                features = data['features']
                return {
                    'continuous_numeric': features.get('continuous_numeric', []),
                    'categorical': features.get('categorical', []),
                    'binary': features.get('binary', []),
                    'all_features': (features.get('continuous_numeric', []) +
                                   features.get('categorical', []) +
                                   features.get('binary', []))
                }

        # Standard feature_lists.json format
        # Support both old and new key names for backward compatibility
        categorical = data.get('categorical') or data.get('categorical_features', [])
        continuous_numeric = data.get('continuous_numeric', [])
        binary = data.get('binary', [])

        # If continuous_features exists but continuous_numeric doesn't, split it
        if not continuous_numeric and 'continuous_features' in data:
            # continuous_features contains both continuous_numeric and binary
            # We need to separate them if possible, otherwise treat all as continuous
            continuous_features = data['continuous_features']
            # Binary features are typically boolean 0/1 flags
            # For now, treat all continuous_features as continuous_numeric
            continuous_numeric = continuous_features

        all_features = data.get('all_features', categorical + continuous_numeric + binary)

        return {
            'continuous_numeric': continuous_numeric,
            'categorical': categorical,
            'binary': binary,
            'all_features': all_features
        }

    @classmethod
    def get_categorical_features(cls) -> List[str]:
        """Get list of categorical features."""
        return cls.load()['categorical']

    @classmethod
    def get_continuous_numeric_features(cls) -> List[str]:
        """Get list of continuous numeric features."""
        return cls.load()['continuous_numeric']

    @classmethod
    def get_binary_features(cls) -> List[str]:
        """Get list of binary features."""
        return cls.load()['binary']

    @classmethod
    def get_all_features(cls) -> List[str]:
        """Get list of all features."""
        return cls.load()['all_features']


class ModelConfig:
    """Configuration for model hyperparameters."""

    DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
    DEFAULT_LOGS_DIR = Path("models/logs")

    # Fallback hyperparameters if no config found
    FALLBACK_XGBOOST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "gamma": 0.6,
        "scale_pos_weight": 8,
        "eval_metric": "aucpr",
    }

    FALLBACK_RANDOM_FOREST_PARAMS = {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    }

    @classmethod
    def load_hyperparameters(
        cls,
        model_type: str = "xgboost",
        source: str = "metadata",
        random_seed: int = None
    ) -> Dict:
        """Load model hyperparameters from configuration.

        Args:
            model_type: Type of model ('xgboost', 'random_forest')
            source: Source to load from ('metadata', 'cv_results', or path to JSON)
            random_seed: Random seed to use (if None, uses value from config or 1)

        Returns:
            Dictionary of hyperparameters

        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If source file doesn't exist
        """
        if model_type.lower() not in ["xgboost", "random_forest"]:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'xgboost' or 'random_forest'")

        params = None

        # Try loading from specified source
        if source == "metadata":
            params = cls._load_from_metadata(model_type)
        elif source == "cv_results":
            params = cls._load_from_cv_results(model_type)
        else:
            # Assume it's a file path
            params = cls._load_from_file(source, model_type)

        # Use fallback if loading failed
        if params is None:
            print(f"  ℹ️  Using fallback {model_type} hyperparameters")
            if model_type.lower() == "xgboost":
                params = cls.FALLBACK_XGBOOST_PARAMS.copy()
            else:
                params = cls.FALLBACK_RANDOM_FOREST_PARAMS.copy()

        # Set random seed
        if random_seed is not None:
            if model_type.lower() == "xgboost":
                params['random_state'] = random_seed
            else:
                params['random_state'] = random_seed

        # Add common parameters for XGBoost
        if model_type.lower() == "xgboost":
            params.setdefault('n_jobs', -1)
            params.setdefault('eval_metric', 'aucpr')

        return params

    @classmethod
    def _load_from_metadata(cls, model_type: str) -> Optional[Dict]:
        """Load hyperparameters from model_metadata.json."""
        if not cls.DEFAULT_METADATA_PATH.exists():
            return None

        try:
            with open(cls.DEFAULT_METADATA_PATH, 'r') as f:
                metadata = json.load(f)

            if 'hyperparameters' in metadata:
                params = metadata['hyperparameters'].copy()
                print(f"  ✓ Loaded {model_type} hyperparameters from {cls.DEFAULT_METADATA_PATH}")
                return params
        except Exception as e:
            print(f"  ⚠️  Failed to load from metadata: {e}")
            return None

        return None

    @classmethod
    def _load_from_cv_results(cls, model_type: str) -> Optional[Dict]:
        """Load best hyperparameters from CV results files."""
        if not cls.DEFAULT_LOGS_DIR.exists():
            return None

        try:
            if model_type.lower() == "random_forest":
                pattern = "random_forest_cv_results_*.csv"
            else:
                pattern = "xgboost_cv_results_*.csv"

            files = sorted(cls.DEFAULT_LOGS_DIR.glob(pattern))
            if not files:
                return None

            # Load most recent CV results
            cv_results = pd.read_csv(files[-1])
            best_row = cv_results.nlargest(1, 'mean_test_score').iloc[0]

            # Extract parameters
            params = {}
            for col in best_row.index:
                if col.startswith('param_classifier__'):
                    param_name = col.replace('param_classifier__', '')
                    value = best_row[col]

                    # Convert types
                    if param_name in ['n_estimators', 'max_depth', 'min_samples_split',
                                     'min_samples_leaf', 'min_child_weight']:
                        params[param_name] = int(value)
                    elif param_name in ['learning_rate', 'subsample', 'colsample_bytree',
                                       'gamma', 'scale_pos_weight', 'reg_alpha', 'reg_lambda']:
                        params[param_name] = float(value)
                    else:
                        params[param_name] = value

            print(f"  ✓ Loaded {model_type} best params from {files[-1].name}")
            return params

        except Exception as e:
            print(f"  ⚠️  Failed to load from CV results: {e}")
            return None

    @classmethod
    def _load_from_file(cls, filepath: str, model_type: str) -> Optional[Dict]:
        """Load hyperparameters from a custom JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Support both direct params dict or nested structure
            if 'hyperparameters' in data:
                params = data['hyperparameters']
            elif model_type in data:
                params = data[model_type]
            else:
                params = data

            print(f"  ✓ Loaded {model_type} hyperparameters from {filepath}")
            return params

        except Exception as e:
            print(f"  ⚠️  Failed to load from {filepath}: {e}")
            return None
