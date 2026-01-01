#!/usr/bin/env python3
"""
Flask API Server for OpenGTO Frontend.

Provides REST endpoints for the Electron frontend to interact with
the trained GTO model.
"""
import sys
import os
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer_interface import GTOTrainerInterface, ACTION_NAMES as TRAINER_ACTION_NAMES
from src.game_state import Position
from src.information_set import ACTION_TYPE_TO_IDX, IDX_TO_ACTION_TYPE, NUM_ACTIONS, get_legal_actions_mask
from src.card import get_all_hand_types

app = Flask(__name__)
CORS(app)

# Global trainer instance
trainer = None

# Store scenarios by ID to handle concurrent requests
scenarios = {}

ACTION_NAMES = ['fold', 'check', 'call', 'bet', 'raise', 'all-in']


def get_trainer():
    """Get or initialize the trainer."""
    global trainer
    if trainer is None:
        # Find best checkpoint
        checkpoint_dirs = ['checkpoints_improved', 'checkpoints']
        checkpoint_path = None

        for cdir in checkpoint_dirs:
            if os.path.exists(cdir):
                files = [f for f in os.listdir(cdir) if f.endswith('.pt')]
                if 'gto_trainer_final.pt' in files:
                    checkpoint_path = os.path.join(cdir, 'gto_trainer_final.pt')
                    break
                elif files:
                    files.sort()
                    checkpoint_path = os.path.join(cdir, files[-1])
                    break

        if checkpoint_path is None:
            raise RuntimeError("No checkpoint found")

        trainer = GTOTrainerInterface(
            checkpoint_path=checkpoint_path,
            num_players=6,
            device='cpu'
        )
        trainer.load_model()
        print(f"Loaded model from {checkpoint_path}")

    return trainer


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


@app.route('/scenario', methods=['GET'])
def get_scenario():
    """Generate a new random scenario."""
    try:
        t = get_trainer()
        scenario = t.generate_random_scenario()

        # Generate unique ID for this scenario
        scenario_id = str(uuid.uuid4())
        scenarios[scenario_id] = scenario

        # Clean up old scenarios (keep last 100)
        if len(scenarios) > 100:
            oldest_keys = list(scenarios.keys())[:-100]
            for key in oldest_keys:
                del scenarios[key]

        # Convert to JSON-serializable format
        response = {
            'scenarioId': scenario_id,
            'heroPosition': scenario.hero_position.name,
            'heroCards': {
                'card1': {
                    'rank': str(scenario.hero_hand.card1.rank),
                    'suit': scenario.hero_hand.card1.suit.name.lower(),
                },
                'card2': {
                    'rank': str(scenario.hero_hand.card2.rank),
                    'suit': scenario.hero_hand.card2.suit.name.lower(),
                },
                'handType': scenario.hero_hand.hand_type_string(),
            },
            'pot': float(scenario.state.pot),
            'currentBet': float(scenario.state.current_bet),
            'stackSize': float(scenario.stack_size),
            'actions': [
                {
                    'position': action.split()[0],
                    'action': 'raise' if 'raises' in action else 'fold' if 'folds' in action else 'call',
                    'amount': float(action.split()[-1].replace('bb', '')) if 'raises' in action else None,
                }
                for action in scenario.villain_actions
            ],
            'legalActions': [a.lower() for a in scenario.legal_actions],
            'players': [
                {
                    'position': pos.name,
                    'stack': float(scenario.stack_size),
                    'isActive': True,
                    'isHero': pos == scenario.hero_position,
                }
                for pos in list(Position)[:6]
            ],
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error generating scenario: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate_action():
    """Evaluate a user's action against GTO."""
    try:
        data = request.json
        user_action = data.get('action', '').lower()
        scenario_id = data.get('scenarioId')

        t = get_trainer()

        # Get scenario by ID
        if not scenario_id or scenario_id not in scenarios:
            return jsonify({'error': 'Invalid or expired scenario ID'}), 400

        scenario = scenarios[scenario_id]

        # Map action to index
        action_map = {
            'fold': 0, 'check': 1, 'call': 2,
            'bet': 3, 'raise': 4, 'all-in': 5
        }
        user_action_idx = action_map.get(user_action)

        if user_action_idx is None:
            return jsonify({'error': f'Invalid action: {user_action}'}), 400

        # Evaluate
        gto_probs, was_correct, ev_loss, feedback = t.evaluate_decision(
            scenario, user_action_idx
        )

        # Format response
        response = {
            'userAction': user_action,
            'gtoStrategy': {
                'fold': float(gto_probs.get('Fold', 0)),
                'check': float(gto_probs.get('Check', 0)),
                'call': float(gto_probs.get('Call', 0)),
                'raise': float(gto_probs.get('Raise', 0)),
                'allIn': float(gto_probs.get('All-In', 0)),
            },
            'isCorrect': bool(was_correct),
            'feedback': feedback,
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error evaluating action: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/scenario/new', methods=['POST'])
def new_scenario():
    """Generate and store a new scenario."""
    try:
        t = get_trainer()
        scenario = t.generate_random_scenario()

        # Generate unique ID for this scenario
        scenario_id = str(uuid.uuid4())
        scenarios[scenario_id] = scenario

        # Clean up old scenarios (keep last 100)
        if len(scenarios) > 100:
            oldest_keys = list(scenarios.keys())[:-100]
            for key in oldest_keys:
                del scenarios[key]

        # Convert to JSON-serializable format
        response = {
            'scenarioId': scenario_id,
            'heroPosition': scenario.hero_position.name,
            'heroCards': {
                'card1': {
                    'rank': str(scenario.hero_hand.card1.rank),
                    'suit': scenario.hero_hand.card1.suit.name.lower(),
                },
                'card2': {
                    'rank': str(scenario.hero_hand.card2.rank),
                    'suit': scenario.hero_hand.card2.suit.name.lower(),
                },
                'handType': scenario.hero_hand.hand_type_string(),
            },
            'pot': float(scenario.state.pot),
            'currentBet': float(scenario.state.current_bet),
            'stackSize': float(scenario.stack_size),
            'actions': [],
            'legalActions': [a.lower() for a in scenario.legal_actions],
            'players': [],
        }

        # Parse villain actions
        for action_str in scenario.villain_actions:
            parts = action_str.split()
            pos = parts[0]
            if 'folds' in action_str:
                response['actions'].append({
                    'position': pos,
                    'action': 'fold',
                })
            elif 'raises' in action_str:
                amount = float(parts[-1].replace('bb', ''))
                response['actions'].append({
                    'position': pos,
                    'action': 'raise',
                    'amount': amount,
                })
            elif 'calls' in action_str:
                response['actions'].append({
                    'position': pos,
                    'action': 'call',
                })

        # Add players
        for pos in list(Position)[:6]:
            response['players'].append({
                'position': pos.name,
                'stack': float(scenario.stack_size),
                'isActive': True,
                'isHero': pos == scenario.hero_position,
            })

        return jsonify(response)

    except Exception as e:
        print(f"Error generating scenario: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/range', methods=['POST'])
def get_range():
    """
    Get GTO strategy for all 169 hand types in a specific scenario.

    Request body:
    {
        "heroPosition": "CO",
        "actions": [
            {"position": "UTG", "action": "fold"},
            {"position": "HJ", "action": "raise", "amount": 2.5}
        ],
        "stack": 100
    }
    """
    try:
        data = request.json
        hero_position = data.get('heroPosition', 'BTN')
        actions = data.get('actions', [])
        stack = data.get('stack', 100.0)

        t = get_trainer()

        # Convert actions to action_history format
        action_history = []
        for action in actions:
            pos = action.get('position', '')
            act = action.get('action', '').lower()
            amount = action.get('amount')

            if act == 'fold':
                action_history.append((pos, 'fold'))
            elif act == 'call':
                action_history.append((pos, 'call'))
            elif act == 'raise':
                amt = amount if amount else 2.5
                action_history.append((pos, f'raise {amt}'))
            elif act == 'all-in':
                action_history.append((pos, f'raise {stack}'))

        # Get all 169 hand types
        all_hands = get_all_hand_types()

        # Query strategy for each hand
        range_data = {}
        legal_actions_set = set()

        # Action totals for distribution
        action_totals = {
            'fold': 0.0,
            'check': 0.0,
            'call': 0.0,
            'raise': 0.0,
            'allIn': 0.0
        }
        hand_count = 0

        for hand_type in all_hands:
            try:
                scenario = t.generate_scenario(
                    position=hero_position,
                    hand=hand_type,
                    stack=stack,
                    action_history=action_history if action_history else None
                )

                # Get GTO strategy
                strategy = t.get_gto_strategy(scenario.state)
                legal_mask = get_legal_actions_mask(scenario.state)

                # Build strategy dict for this hand
                hand_strategy = {}
                for i in range(NUM_ACTIONS):
                    if legal_mask[i]:
                        action_name = TRAINER_ACTION_NAMES[i]
                        prob = float(strategy[i])
                        # Map action names to frontend format
                        if action_name == 'Fold':
                            hand_strategy['fold'] = prob
                            action_totals['fold'] += prob
                        elif action_name == 'Check':
                            hand_strategy['check'] = prob
                            action_totals['check'] += prob
                        elif action_name == 'Call':
                            hand_strategy['call'] = prob
                            action_totals['call'] += prob
                        elif action_name == 'Raise':
                            hand_strategy['raise'] = prob
                            action_totals['raise'] += prob
                        elif action_name == 'All-In':
                            hand_strategy['allIn'] = prob
                            action_totals['allIn'] += prob
                        legal_actions_set.add(action_name.lower().replace('-', ''))

                range_data[hand_type] = hand_strategy
                hand_count += 1

            except Exception as e:
                # Skip hands that can't be processed
                print(f"Warning: Could not process hand {hand_type}: {e}")
                range_data[hand_type] = {'fold': 1.0}

        # Calculate action distribution (average across all hands)
        action_distribution = {}
        if hand_count > 0:
            for action, total in action_totals.items():
                action_distribution[action] = total / hand_count

        response = {
            'heroPosition': hero_position,
            'rangeData': range_data,
            'legalActions': list(legal_actions_set),
            'actionDistribution': action_distribution,
            'handCount': hand_count
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error getting range: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/range/hand', methods=['POST'])
def get_hand_strategy():
    """
    Get GTO strategy for a single hand in a specific scenario.
    Faster than /range for single hand queries.
    """
    try:
        data = request.json
        hero_position = data.get('heroPosition', 'BTN')
        hand = data.get('hand', 'AA')
        actions = data.get('actions', [])
        stack = data.get('stack', 100.0)

        t = get_trainer()

        # Convert actions to action_history format
        action_history = []
        for action in actions:
            pos = action.get('position', '')
            act = action.get('action', '').lower()
            amount = action.get('amount')

            if act == 'fold':
                action_history.append((pos, 'fold'))
            elif act == 'call':
                action_history.append((pos, 'call'))
            elif act == 'raise':
                amt = amount if amount else 2.5
                action_history.append((pos, f'raise {amt}'))
            elif act == 'all-in':
                action_history.append((pos, f'raise {stack}'))

        scenario = t.generate_scenario(
            position=hero_position,
            hand=hand,
            stack=stack,
            action_history=action_history if action_history else None
        )

        # Get GTO strategy
        strategy = t.get_gto_strategy(scenario.state)
        legal_mask = get_legal_actions_mask(scenario.state)

        # Build strategy dict
        hand_strategy = {}
        for i in range(NUM_ACTIONS):
            if legal_mask[i]:
                action_name = TRAINER_ACTION_NAMES[i]
                prob = float(strategy[i])
                if action_name == 'Fold':
                    hand_strategy['fold'] = prob
                elif action_name == 'Check':
                    hand_strategy['check'] = prob
                elif action_name == 'Call':
                    hand_strategy['call'] = prob
                elif action_name == 'Raise':
                    hand_strategy['raise'] = prob
                elif action_name == 'All-In':
                    hand_strategy['allIn'] = prob

        response = {
            'hand': hand,
            'heroPosition': hero_position,
            'strategy': hand_strategy,
            'pot': float(scenario.state.pot),
            'currentBet': float(scenario.state.current_bet)
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error getting hand strategy: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting OpenGTO API Server...")
    print("Loading model...")

    try:
        get_trainer()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Server will start but may return errors until model is available.")

    print("\nAPI Server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
