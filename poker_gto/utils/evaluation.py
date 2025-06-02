# poker_gto/utils/evaluation.py
"""
Model evaluation utilities.
"""

import torch
import logging
from typing import List, Dict

def evaluate_model(model, test_scenarios: List[Dict]) -> Dict:
    """Evaluate model performance on test scenarios."""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model performance...")
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    action_performance = {'fold': {'correct': 0, 'total': 0},
                         'call': {'correct': 0, 'total': 0},
                         'raise': {'correct': 0, 'total': 0}}
    
    with torch.no_grad():
        for scenario in test_scenarios:
            features = scenario['features']
            true_action = scenario['optimal_action']['action']
            
            # Get model prediction
            predicted_action, predicted_size = model.predict_action(features)
            
            # Check if prediction is correct
            is_correct = predicted_action == true_action
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            # Track per-action performance
            if true_action in action_performance:
                action_performance[true_action]['total'] += 1
                if is_correct:
                    action_performance[true_action]['correct'] += 1
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_predictions
    
    action_accuracies = {}
    for action in action_performance:
        if action_performance[action]['total'] > 0:
            action_accuracies[action] = action_performance[action]['correct'] / action_performance[action]['total']
        else:
            action_accuracies[action] = 0.0
    
    results = {
        'overall_accuracy': overall_accuracy,
        'action_accuracies': action_accuracies,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions
    }
    
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    for action, accuracy in action_accuracies.items():
        logger.info(f"{action.capitalize()} Accuracy: {accuracy:.4f}")
    
    return results
