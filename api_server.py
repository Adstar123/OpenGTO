#!/usr/bin/env python3
"""
Flask API Server for OpenGTO Frontend.

Provides REST endpoints for the Electron frontend to interact with
the trained GTO model.
"""
import sys
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer_interface import GTOTrainerInterface
from src.game_state import Position
from src.information_set import ACTION_TYPE_TO_IDX, IDX_TO_ACTION_TYPE, NUM_ACTIONS

app = Flask(__name__)
CORS(app)

# Global trainer instance
trainer = None

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
        t._current_scenario = scenario  # Store for evaluate endpoint

        # Convert to JSON-serializable format
        response = {
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
            'legalActions': scenario.legal_actions,
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

        t = get_trainer()

        # Get current scenario state from trainer
        if not hasattr(t, '_current_scenario') or t._current_scenario is None:
            return jsonify({'error': 'No active scenario'}), 400

        scenario = t._current_scenario

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
            'isCorrect': was_correct,
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
        t._current_scenario = scenario

        # Convert to JSON-serializable format
        response = {
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
