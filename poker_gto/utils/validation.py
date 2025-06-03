"""Data validation utilities for ensuring data quality."""

from typing import Dict, List, Any, Optional, Set
import numpy as np
from collections import Counter
import logging


class DataValidator:
    """Validates training data for quality and consistency."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize validator.
        
        Args:
            logger: Logger for validation messages
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_scenario(self, scenario: Dict[str, Any]) -> List[str]:
        """Validate a single scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ['features', 'optimal_action', 'context']
        for field in required_fields:
            if field not in scenario:
                errors.append(f"Missing required field: {field}")
        
        if 'features' in scenario:
            feature_errors = self._validate_features(scenario['features'])
            errors.extend(feature_errors)
        
        if 'optimal_action' in scenario:
            action_errors = self._validate_action(scenario['optimal_action'])
            errors.extend(action_errors)
        
        return errors
    
    def validate_dataset(
        self,
        scenarios: List[Dict[str, Any]],
        expected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate an entire dataset.
        
        Args:
            scenarios: List of scenarios
            expected_features: Expected feature names
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_scenarios': len(scenarios),
            'valid_scenarios': 0,
            'invalid_scenarios': 0,
            'errors': [],
            'action_distribution': Counter(),
            'feature_statistics': {},
            'warnings': []
        }
        
        all_features = set()
        feature_values = {}
        
        for i, scenario in enumerate(scenarios):
            errors = self.validate_scenario(scenario)
            
            if errors:
                report['invalid_scenarios'] += 1
                report['errors'].append({
                    'scenario_index': i,
                    'errors': errors
                })
            else:
                report['valid_scenarios'] += 1
                
                # Track action distribution
                action = scenario['optimal_action']['action']
                report['action_distribution'][action] += 1
                
                # Track features
                features = scenario['features']
                all_features.update(features.keys())
                
                for feature, value in features.items():
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(value)
        
        # Check feature consistency
        if expected_features:
            missing_features = set(expected_features) - all_features
            extra_features = all_features - set(expected_features)
            
            if missing_features:
                report['warnings'].append(
                    f"Missing expected features: {missing_features}"
                )
            if extra_features:
                report['warnings'].append(
                    f"Unexpected features found: {extra_features}"
                )
        
        # Calculate feature statistics
        for feature, values in feature_values.items():
            values_array = np.array(values)
            report['feature_statistics'][feature] = {
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'unique_values': len(np.unique(values_array))
            }
        
        # Check action balance
        action_counts = report['action_distribution']
        if action_counts:
            min_count = min(action_counts.values())
            max_count = max(action_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 2.0:
                report['warnings'].append(
                    f"Action distribution is imbalanced (ratio: {imbalance_ratio:.2f})"
                )
        
        # Log summary
        self.logger.info(f"Dataset validation complete:")
        self.logger.info(f"  Total scenarios: {report['total_scenarios']}")
        self.logger.info(f"  Valid: {report['valid_scenarios']}")
        self.logger.info(f"  Invalid: {report['invalid_scenarios']}")
        self.logger.info(f"  Action distribution: {dict(action_counts)}")
        
        if report['warnings']:
            for warning in report['warnings']:
                self.logger.warning(f"  {warning}")
        
        return report
    
    def _validate_features(self, features: Dict[str, float]) -> List[str]:
        """Validate feature dictionary."""
        errors = []
        
        if not isinstance(features, dict):
            errors.append("Features must be a dictionary")
            return errors
        
        for key, value in features.items():
            if not isinstance(key, str):
                errors.append(f"Feature key must be string, got {type(key)}")
            
            if not isinstance(value, (int, float)):
                errors.append(f"Feature '{key}' must be numeric, got {type(value)}")
            elif not np.isfinite(value):
                errors.append(f"Feature '{key}' has non-finite value: {value}")
            
            # Check common feature ranges
            if key.startswith('position_') and value not in [0.0, 1.0]:
                errors.append(f"Position feature '{key}' should be 0 or 1, got {value}")
            
            if key in ['pot_odds', 'hand_strength', 'position_strength']:
                if not 0.0 <= value <= 1.0:
                    errors.append(f"Feature '{key}' should be in [0, 1], got {value}")
        
        return errors
    
    def _validate_action(self, action: Dict[str, Any]) -> List[str]:
        """Validate action dictionary."""
        errors = []
        
        if not isinstance(action, dict):
            errors.append("Action must be a dictionary")
            return errors
        
        if 'action' not in action:
            errors.append("Action dictionary missing 'action' field")
        elif action['action'] not in ['fold', 'call', 'raise']:
            errors.append(f"Invalid action: {action['action']}")
        
        if 'size' in action:
            size = action['size']
            if not isinstance(size, (int, float)):
                errors.append(f"Action size must be numeric, got {type(size)}")
            elif size < 0:
                errors.append(f"Action size cannot be negative: {size}")
            elif action['action'] == 'raise' and size == 0:
                errors.append("Raise action must have non-zero size")
        
        return errors


class FeatureValidator:
    """Validates feature extraction consistency."""
    
    @staticmethod
    def validate_feature_names(
        features: Dict[str, float],
        expected_names: List[str]
    ) -> List[str]:
        """Validate that features match expected names.
        
        Args:
            features: Feature dictionary
            expected_names: Expected feature names
            
        Returns:
            List of validation errors
        """
        errors = []
        
        feature_set = set(features.keys())
        expected_set = set(expected_names)
        
        missing = expected_set - feature_set
        if missing:
            errors.append(f"Missing features: {missing}")
        
        extra = feature_set - expected_set
        if extra:
            errors.append(f"Unexpected features: {extra}")
        
        return errors
    
    @staticmethod
    def validate_feature_ranges(features: Dict[str, float]) -> List[str]:
        """Validate that features are in expected ranges.
        
        Args:
            features: Feature dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Define expected ranges for common features
        expected_ranges = {
            'pot_odds': (0.0, 1.0),
            'hand_strength': (0.0, 1.0),
            'position_strength': (0.0, 1.0),
            'stack_ratio': (0.0, float('inf')),
            'pot_size': (0.0, float('inf')),
            'bet_to_call': (0.0, float('inf')),
        }
        
        for feature, value in features.items():
            # Check position features
            if feature.startswith('position_'):
                if value not in [0.0, 1.0]:
                    errors.append(
                        f"Position feature '{feature}' should be 0 or 1, got {value}"
                    )
            
            # Check binary features
            elif feature in ['is_pocket_pair', 'is_suited', 'facing_raise',
                           'premium_hand', 'strong_hand', 'playable_hand']:
                if value not in [0.0, 1.0]:
                    errors.append(
                        f"Binary feature '{feature}' should be 0 or 1, got {value}"
                    )
            
            # Check range features
            elif feature in expected_ranges:
                min_val, max_val = expected_ranges[feature]
                if not min_val <= value <= max_val:
                    errors.append(
                        f"Feature '{feature}' out of range [{min_val}, {max_val}]: {value}"
                    )
        
        return errors