"""Unit tests for preflop model."""

import unittest
import torch
import tempfile
from pathlib import Path

from poker_gto.ml.models.preflop_model import PreflopGTOModel, ModelConfig
from poker_gto.ml.features.feature_extractors import PreflopFeatureExtractor


class TestPreflopModel(unittest.TestCase):
    """Test cases for PreflopGTOModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            input_size=20,
            hidden_sizes=[64, 32],
            dropout_rate=0.1
        )
        self.model = PreflopGTOModel(self.config)
        self.extractor = PreflopFeatureExtractor()
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.config.input_size, 20)
        self.assertEqual(self.model.config.hidden_sizes, [64, 32])
        self.assertEqual(self.model.config.output_size, 3)
        
        # Check network structure
        modules = list(self.model.network.modules())
        linear_layers = [m for m in modules if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 3)  # 20->64, 64->32, 32->3
    
    def test_forward_pass(self):
        """Test forward pass works correctly."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 20)
        
        output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 3))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_feature_conversion(self):
        """Test feature dictionary to tensor conversion."""
        # Create sample features
        features = {f: 0.0 for f in PreflopGTOModel.FEATURE_NAMES}
        features['position_BTN'] = 1.0
        features['hand_strength'] = 0.7
        
        tensor = self.model.features_to_tensor(features)
        
        self.assertEqual(tensor.shape, (20,))
        self.assertEqual(tensor[3].item(), 1.0)  # position_BTN is at index 3
    
    def test_missing_features_error(self):
        """Test error on missing features."""
        incomplete_features = {'position_BTN': 1.0}  # Missing most features
        
        with self.assertRaises(ValueError) as context:
            self.model.features_to_tensor(incomplete_features)
        
        self.assertIn("Missing required features", str(context.exception))
    
    def test_predict_action(self):
        """Test action prediction."""
        # Create valid features
        scenario = {
            'position': 'BTN',
            'hole_cards': 'AKs',
            'facing_raise': False,
            'pot_size': 1.5,
            'bet_to_call': 0.0,
            'stack_ratio': 1.0,
            'num_players': 6
        }
        
        features = self.extractor.extract_from_scenario(scenario)
        action, size = self.model.predict_action(features)
        
        self.assertIn(action, ['fold', 'call', 'raise'])
        if action == 'raise':
            self.assertGreater(size, 0.0)
        else:
            self.assertEqual(size, 0.0)
    
    def test_action_probabilities(self):
        """Test getting action probability distribution."""
        scenario = {
            'position': 'UTG',
            'hole_cards': '72o',
            'facing_raise': True,
            'pot_size': 5.0,
            'bet_to_call': 3.0,
            'stack_ratio': 1.0,
            'num_players': 6
        }
        
        features = self.extractor.extract_from_scenario(scenario)
        probs = self.model.get_action_probabilities(features)
        
        self.assertSetEqual(set(probs.keys()), {'fold', 'call', 'raise'})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        
        # For 72o facing raise from UTG, fold should be highest probability
        # (though untrained model might not reflect this)
        for action, prob in probs.items():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pth"
            
            # Save model
            metadata = {'test': True, 'version': '1.0'}
            self.model.save(str(filepath), metadata)
            
            self.assertTrue(filepath.exists())
            
            # Load model
            loaded_model = PreflopGTOModel.load(str(filepath))
            
            # Check loaded model matches original
            self.assertEqual(
                loaded_model.config.input_size,
                self.model.config.input_size
            )
            self.assertEqual(
                loaded_model.config.hidden_sizes,
                self.model.config.hidden_sizes
            )
            
            # Check weights match
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Wrong input size should raise error
        bad_config = ModelConfig(
            input_size=15,  # Wrong size
            hidden_sizes=[64, 32]
        )
        
        with self.assertRaises(ValueError) as context:
            PreflopGTOModel(bad_config)
        
        self.assertIn("doesn't match expected features", str(context.exception))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model with other components."""
    
    def test_model_with_feature_extractor(self):
        """Test model works with feature extractor."""
        model = PreflopGTOModel()
        extractor = PreflopFeatureExtractor()
        
        # Test various scenarios
        scenarios = [
            {'position': 'BTN', 'hole_cards': 'AA', 'facing_raise': False},
            {'position': 'UTG', 'hole_cards': '72o', 'facing_raise': True},
            {'position': 'BB', 'hole_cards': 'KQs', 'facing_raise': False},
        ]
        
        for scenario_data in scenarios:
            # Add required fields
            scenario_data.update({
                'pot_size': 1.5,
                'bet_to_call': 0.0,
                'stack_ratio': 1.0,
                'num_players': 6,
                'player_count': 6
            })
            
            features = extractor.extract_from_scenario(scenario_data)
            action, size = model.predict_action(features)
            
            self.assertIn(action, ['fold', 'call', 'raise'])


if __name__ == '__main__':
    unittest.main()