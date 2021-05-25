from model import ZeroTTT

# model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', lr=3e-4, board_len=10)
model = ZeroTTT(brain_path=None, opt_path=None, lr=0.02, board_len=10)
print(model.get_parameter_count())

quit()

# test = torch.randn((2, 10, 10))
# p, v = model.predict(test)
# print(v.shape)

model.self_play(n_games=1000, num_simulations=200, render=1, positions_per_learn=100000, batch_size=40)
# model.self_play(n_games=1000, num_simulations=10, render=1, positions_per_learn=800, batch_size=40,
            # games_per_evaluation=50, evaluation_game_count=20, evaluation_num_simulations=10)

# model.evaluate(opp_name='best_model', opp_opt_state='best_opt_state', board_len=10, num_simulations=100, render=True, model_token=1)


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


