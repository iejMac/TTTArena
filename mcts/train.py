from model import ZeroTTT

# model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', lr=3e-4, weight_decay=2e-4, board_len=10)
model = ZeroTTT(brain_path=None, opt_path=None, lr=3e-4, weight_decay=2e-4, board_len=10)

print(model.get_parameter_count())
print(sum(p.numel() for p in model.brain.policy_head.parameters() if p.requires_grad))
print(sum(p.numel() for p in model.brain.value_head.parameters() if p.requires_grad))

# model.self_play(n_games=1000, num_simulations=200, render=1, training_epochs=1, min_positions_learn=300000, positions_per_learn=100000, batch_size=40)
model.self_play(n_games=1000, num_simulations=30, render=1, training_epochs=1, min_positions_learn=300, positions_per_learn=100, batch_size=40)

'''
  Bugs:
- If you're traversing into a node that has no open tiles just return the value (no need to go in the node) (currently this might just evaluate it for no reason)
'''

'''
  Improvements:
- Consider changing Node.children to a dict from action to Edge
'''

'''
  Keep eye on:
- Value net saturating since it uses a tanh but nothing reverses the exp on tanh because we use MSE (could saturate)
'''


