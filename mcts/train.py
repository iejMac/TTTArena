from model import ZeroTTT

model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', lr=3e-4, board_len=10)
# model = ZeroTTT(brain_path=None, opt_path=None, lr=0.02, board_len=10)

# test = torch.randn((2, 10, 10))
# p, v = model.predict(test)
# print(v.shape)

model.self_play(n_games=1000, num_simulations=200, render=1, positions_per_learn=100000, batch_size=40)
# model.self_play(n_games=1000, num_simulations=10, render=1, positions_per_learn=800, batch_size=40,
            # games_per_evaluation=50, evaluation_game_count=20, evaluation_num_simulations=10)

# model.evaluate(opp_name='best_model', opp_opt_state='best_opt_state', board_len=10, num_simulations=100, render=True, model_token=1)


'''
  Bugs:
1. Reset search tree or remove child nodes that are chosen!!! (or not?)
2. Play around with UCB (def dont have this aspect right, re-read the papers)
3. Is there really a need to flip the tokens on the board so that current player is always on top? (Probably makes it simpler but idk)
4. Consider changing Node.children to a dict from action to Edge
'''

'''
  Keep eye on:

1. Value net saturating since it uses a tanh but nothing reverses the exp on tanh because we use MSE (could saturate)

'''


