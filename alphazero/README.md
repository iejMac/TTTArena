# ZeroTTT 🧠: 
This appraoch implements the AlphaZero algorithm for this game. ([Link to Paper](https://arxiv.org/abs/1712.01815))

## Results 📊:

## Files 📁:
1. model.py - Model class along with neural network,
2. mcts.py - MCTS class the model uses to perform Monte-Carlo rollouts guided by it's networks,
3. trainer.py - Trainer class which performs self-play games, trains the network, and generates the replay buffer,
4. manager.py - Manager class that manages parallel Trainers for faster replay buffer generation,
5. testing.py - Test class for testing raw network evaluations on pre-set positions and human games,
6. database.py - DataBase class for managing saving, loading, and augmenting data to the replay buffer,
7. agent.py - Agent class for usage in game.py,
8. models - repository with trained models
