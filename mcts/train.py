from model import ZeroTTT

# model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', lr=3e-4, board_len=10)
model = ZeroTTT(brain_path=None, opt_path=None, lr=3e-4, board_len=10)

# test = torch.randn((2, 10, 10))
# p, v = model.predict(test)
# print(v.shape)

model.self_play(n_games=1000, num_simulations=200, render=20, positions_per_learn=1600, batch_size=40,
            games_per_evaluation=50, evaluation_game_count=20, evaluation_num_simulations=200)
# model.self_play(n_games=1000, num_simulations=10, render=1, positions_per_learn=800, batch_size=40,
            # games_per_evaluation=50, evaluation_game_count=20, evaluation_num_simulations=10)

# model.evaluate(opp_name='best_model', opp_opt_state='best_opt_state', board_len=10, num_simulations=100, render=True, model_token=1)
