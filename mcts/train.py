from model import ZeroTTT

# model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', lr=3e-4, weight_decay=2e-4, board_len=10)
model = ZeroTTT(brain_path=None, opt_path=None, lr=3e-4, board_len=10)

model.self_play(n_games=1, num_simulations=2000, render=1200, training_epochs=1, max_position_storage=60000, positions_per_learn=10000, batch_size=40)
# model.self_play(n_games=1000, num_simulations=100, render=3, training_epochs=1, max_position_storage=300, positions_per_learn=100, batch_size=40)



'''
  Bugs:
- If you're traversing into a node that has no open tiles just return the value (no need to go in the node) (currently this might just evaluate it for no reason)
  TODO:
  1. Action space: add pass move which is to be played at the terminal state
  2. Consider adding T time steps in the past
  3. num_simulations should be an input to search, not the construtor of MCTS
'''

'''
  Improvements:
- Consider changing Node.children to a dict from action to Edge
'''

'''
  Keep eye on:
- Value net saturating since it uses a tanh but nothing reverses the exp on tanh because we use MSE (could saturate)
'''


