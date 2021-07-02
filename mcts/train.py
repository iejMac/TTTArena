from model import ZeroTTT

model = ZeroTTT(brain_path="trained_model_0", opt_path="trained_opt_state_0", lr=3e-4, board_len=10)

# model.self_play(n_games=1000, num_simulations=100, render=1, training_epochs=1, max_position_storage=300000, positions_per_learn=100000, batch_size=40)
# model.self_play(n_games=1000, num_simulations=100, render=10, training_epochs=1, max_position_storage=3000, positions_per_learn=1000, batch_size=100)
model.self_play(n_games=1000, num_simulations=200, render=1200, training_epochs=0, max_position_storage=10000, positions_per_learn=30000, batch_size=40, generate_buffer_path="/storage/replay_buffer")

'''
  Bugs:
- If you're traversing into a node that has no open tiles just return the value (no need to go in the node) (currently this might just evaluate it for no reason)
  TODO:
  1. Action space: add pass move which is to be played at the terminal state
  2. Consider adding T time steps in the past
'''

'''
  Improvements:
- Consider changing Node.children to a dict from action to Edge
'''

'''
  Keep eye on:
- Value net saturating since it uses a tanh but nothing reverses the exp on tanh because we use MSE (could saturate)
'''


