"""
Tests for src/deployment/config/artifact_builders.py

Tests shared artifact builders and configuration constants loaded from deployment_defaults.json.
"""

import pytest
from src.deployment.config.artifact_builders import (
    RISK_LEVELS,
    DEFAULT_THRESHOLD,
    MODEL_INFO_DEFAULTS,
    PREPROCESSING_INFO,
    build_threshold_config,
    build_model_metadata,
)


class TestConfigConstants:
    """Tests for configuration constants loaded from deployment_defaults.json"""

    def test_risk_levels_structure(self):
        """Test RISK_LEVELS has expected structure"""
        assert isinstance(RISK_LEVELS, dict)
        assert 'low' in RISK_LEVELS
        assert 'medium' in RISK_LEVELS
        assert 'high' in RISK_LEVELS

    def test_risk_levels_max_probabilities(self):
        """Test RISK_LEVELS have correct max_probability values"""
        assert RISK_LEVELS['low']['max_probability'] == 0.3
        assert RISK_LEVELS['medium']['max_probability'] == 0.7
        assert RISK_LEVELS['high']['max_probability'] == 1.0

    def test_risk_levels_ascending_order(self):
        """Test that risk levels are in ascending order"""
        assert (
            RISK_LEVELS['low']['max_probability']
            < RISK_LEVELS['medium']['max_probability']
            < RISK_LEVELS['high']['max_probability']
        )

    def test_default_threshold_value(self):
        """Test DEFAULT_THRESHOLD has expected value"""
        assert DEFAULT_THRESHOLD == 0.5
        assert isinstance(DEFAULT_THRESHOLD, float)

    def test_model_info_defaults_structure(self):
        """Test MODEL_INFO_DEFAULTS has expected keys"""
        assert isinstance(MODEL_INFO_DEFAULTS, dict)
        assert 'model_name' in MODEL_INFO_DEFAULTS
        assert 'model_type' in MODEL_INFO_DEFAULTS
        assert 'version' in MODEL_INFO_DEFAULTS
        assert 'framework' in MODEL_INFO_DEFAULTS
        assert 'python_version' in MODEL_INFO_DEFAULTS

    def test_model_info_defaults_values(self):
        """Test MODEL_INFO_DEFAULTS has expected values"""
        assert MODEL_INFO_DEFAULTS['model_name'] == 'XGBoost Fraud Detector'
        assert MODEL_INFO_DEFAULTS['model_type'] == 'XGBClassifier'
        assert MODEL_INFO_DEFAULTS['version'] == '1.0'

    def test_preprocessing_info_structure(self):
        """Test PREPROCESSING_INFO has expected keys"""
        assert isinstance(PREPROCESSING_INFO, dict)
        assert 'categorical_encoding' in PREPROCESSING_INFO
        assert 'numeric_scaling' in PREPROCESSING_INFO
        assert 'binary_features' in PREPROCESSING_INFO


class TestBuildThresholdConfig:
    """Tests for build_threshold_config function"""

    @pytest.fixture
    def sample_optimized_thresholds(self):
        """Sample optimized thresholds for testing"""
        return {
            'optimal_f1': {
                'threshold': 0.35,
                'precision': 0.75,
                'recall': 0.80,
                'f1': 0.77,
                'description': 'Optimal F1 score'
            },
            'target_performance': {
                'threshold': 0.30,
                'precision': 0.70,
                'recall': 0.85,
                'f1': 0.76,
                'min_precision': 0.70,
                'description': 'Target performance threshold'
            }
        }

    def test_build_threshold_config_structure(self, sample_optimized_thresholds):
        """Test build_threshold_config returns expected structure"""
        config = build_threshold_config(sample_optimized_thresholds)

        assert 'default_threshold' in config
        assert 'recommended_threshold' in config
        assert 'risk_levels' in config
        assert 'optimized_thresholds' in config
        assert 'note' in config

    def test_build_threshold_config_default_threshold(self, sample_optimized_thresholds):
        """Test default_threshold is set from DEFAULT_THRESHOLD constant"""
        config = build_threshold_config(sample_optimized_thresholds)
        assert config['default_threshold'] == DEFAULT_THRESHOLD

    def test_build_threshold_config_risk_levels(self, sample_optimized_thresholds):
        """Test risk_levels matches RISK_LEVELS constant"""
        config = build_threshold_config(sample_optimized_thresholds)
        assert config['risk_levels'] == RISK_LEVELS

    def test_build_threshold_config_recommended_target_performance(self, sample_optimized_thresholds):
        """Test recommended_threshold is 'target_performance' when present"""
        config = build_threshold_config(sample_optimized_thresholds)
        assert config['recommended_threshold'] == 'target_performance'

    def test_build_threshold_config_recommended_optimal_f1(self):
        """Test recommended_threshold falls back to 'optimal_f1' when target_performance absent"""
        thresholds = {
            'optimal_f1': {
                'threshold': 0.35,
                'precision': 0.75,
                'recall': 0.80,
                'f1': 0.77,
                'description': 'Optimal F1 score'
            }
        }
        config = build_threshold_config(thresholds)
        assert config['recommended_threshold'] == 'optimal_f1'

    def test_build_threshold_config_custom_note(self, sample_optimized_thresholds):
        """Test custom note parameter"""
        custom_note = "Custom note for testing"
        config = build_threshold_config(sample_optimized_thresholds, note=custom_note)
        assert config['note'] == custom_note

    def test_build_threshold_config_preserves_thresholds(self, sample_optimized_thresholds):
        """Test that optimized_thresholds are preserved as-is"""
        config = build_threshold_config(sample_optimized_thresholds)
        assert config['optimized_thresholds'] == sample_optimized_thresholds


class TestBuildModelMetadata:
    """Tests for build_model_metadata function"""

    @pytest.fixture
    def sample_hyperparameters(self):
        """Sample hyperparameters for testing"""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': 10,
            'eval_metric': 'aucpr',
            'random_state': 1
        }

    @pytest.fixture
    def sample_test_metrics(self):
        """Sample test metrics for testing"""
        return {
            'roc_auc': 0.95,
            'pr_auc': 0.85,
            'f1': 0.77,
            'precision': 0.75,
            'recall': 0.80,
            'accuracy': 0.92
        }

    @pytest.fixture
    def sample_dataset_info(self):
        """Sample dataset info for testing"""
        return {
            'training_samples': 80000,
            'test_samples': 20000,
            'num_features': 30,
            'fraud_rate_train': 0.10,
            'fraud_rate_test': 0.10,
            'class_imbalance_ratio': 9.0
        }

    @pytest.fixture
    def sample_feature_lists(self):
        """Sample feature lists for testing"""
        return {
            'continuous_numeric': ['feature1', 'feature2'],
            'categorical': ['category1'],
            'binary': ['binary1', 'binary2', 'binary3']
        }

    def test_build_model_metadata_structure(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test build_model_metadata returns expected structure"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        assert 'model_info' in metadata
        assert 'hyperparameters' in metadata
        assert 'dataset_info' in metadata
        assert 'performance' in metadata
        assert 'features' in metadata
        assert 'preprocessing' in metadata

    def test_build_model_metadata_model_info(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test model_info uses MODEL_INFO_DEFAULTS"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        assert metadata['model_info']['model_name'] == MODEL_INFO_DEFAULTS['model_name']
        assert metadata['model_info']['model_type'] == MODEL_INFO_DEFAULTS['model_type']
        assert metadata['model_info']['version'] == MODEL_INFO_DEFAULTS['version']
        assert 'training_date' in metadata['model_info']

    def test_build_model_metadata_preprocessing(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test preprocessing uses PREPROCESSING_INFO"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        assert metadata['preprocessing'] == PREPROCESSING_INFO

    def test_build_model_metadata_test_set_metrics(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test test_set metrics are properly included"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        test_set = metadata['performance']['test_set']
        assert test_set['roc_auc'] == sample_test_metrics['roc_auc']
        assert test_set['pr_auc'] == sample_test_metrics['pr_auc']
        assert test_set['f1_score'] == sample_test_metrics['f1']
        assert test_set['precision'] == sample_test_metrics['precision']
        assert test_set['recall'] == sample_test_metrics['recall']
        assert test_set['accuracy'] == sample_test_metrics['accuracy']

    def test_build_model_metadata_feature_count(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test total_count is calculated from feature lists"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        expected_count = (
            len(sample_feature_lists['continuous_numeric'])
            + len(sample_feature_lists['categorical'])
            + len(sample_feature_lists['binary'])
        )
        assert metadata['features']['total_count'] == expected_count

    def test_build_model_metadata_custom_note(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test custom note parameter"""
        custom_note = "Custom model note"
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists,
            note=custom_note
        )

        assert metadata['model_info']['note'] == custom_note

    def test_build_model_metadata_with_cv_metrics(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test cv_metrics are included when provided"""
        cv_metrics = {
            'cv_folds': 4,
            'cv_strategy': 'StratifiedKFold',
            'cv_pr_auc': 0.84,
            'note': 'CV performed on train set'
        }
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists,
            cv_metrics=cv_metrics
        )

        assert 'cross_validation' in metadata['performance']
        assert metadata['performance']['cross_validation'] == cv_metrics

    def test_build_model_metadata_without_cv_metrics(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test cv_metrics are not included when not provided"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        assert 'cross_validation' not in metadata['performance']

    def test_build_model_metadata_with_workflow_info(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test workflow_info is included when provided"""
        workflow_info = {
            'training_notebook': 'fd2_model_selection_tuning.ipynb',
            'evaluation_notebook': 'fd3_model_evaluation_deployment.ipynb'
        }
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists,
            workflow_info=workflow_info
        )

        assert 'workflow' in metadata
        assert metadata['workflow'] == workflow_info

    def test_build_model_metadata_without_workflow_info(
        self, sample_hyperparameters, sample_test_metrics, sample_dataset_info, sample_feature_lists
    ):
        """Test workflow is not included when not provided"""
        metadata = build_model_metadata(
            hyperparameters=sample_hyperparameters,
            test_metrics=sample_test_metrics,
            dataset_info=sample_dataset_info,
            feature_lists=sample_feature_lists
        )

        assert 'workflow' not in metadata
