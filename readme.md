# OpenGTO

A preflop poker trainer that uses neural networks trained with Deep Counterfactual Regret Minimisation to teach Game Theory Optimal (GTO) play.

## Features

- **GTO Strategy Training**: Learn optimal preflop decisions based on position and opponent actions
- **Range Viewer**: Visualise opening and response ranges with a 13x13 hand matrix showing action frequencies
- **Interactive Scenarios**: Practice with realistic preflop situations including opens, 3-bets, 4-bets, and 5-bets
- **Real-time Feedback**: See the GTO-recommended action and compare it with your choice
- **Progress Tracking**: Monitor your accuracy over time with session statistics

## How It Works

OpenGTO uses a neural network trained through self-play using Deep CFR (Counterfactual Regret Minimisation). The model learns optimal strategies by playing millions of hands against itself, gradually converging towards game theory optimal play.

When you practice:
1. A random preflop scenario is generated based on your selected position and opponent actions
2. You choose your action (fold, call, or raise)
3. The neural network evaluates the scenario and provides the GTO-recommended action
4. Your response is compared against the optimal play

## Installation

1. Download the latest `OpenGTO-*-win.zip` from the [Releases](https://github.com/Adstar123/OpenGTO/releases) page
2. Extract the zip file to your preferred location
3. Run `OpenGTO.exe` from the extracted folder

### Windows Firewall Notice

On first launch, Windows Firewall may ask whether to allow the backend to access devices on your network. This occurs because the backend server binds to `0.0.0.0` (all network interfaces) when starting the local Flask server. The application only communicates locally between the frontend and backend - it does not send any data over the internet. You can safely allow or deny network access; the application will function correctly either way as it only requires localhost communication.

## System Requirements

- Windows 10/11 (64-bit)
- 4GB RAM minimum
- 3GB disk space

