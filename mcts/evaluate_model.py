import os
import numpy as np

from model import ZeroTTT
from environment import prepare_state
from environment import Environment

'''
This scirpt calculates a metric that shows if the model thinks human moves are good.
Assuming consistent improvement and that human level ability is between random play and super-human ability, we should be able to obvserve an increase in this metric as the model approaches human ability followed by a smaller decrease when it surpasses human ability.
'''

board_len = 10
data_dir = "../data/30x30"
games = os.listdir(data_dir)
compatible_games = []

model = ZeroTTT(brain_path='best_model', opt_path='best_opt_state', board_len=board_len)
env = Environment(board_len=board_len)

human_metric = 0.0
value_asymmetry = 0.0

for game in games:
  game_hist = np.loadtxt(os.path.join(data_dir, game), delimiter=",", dtype=int)

  # Check if game is compatible with board_len
  cluster_width = np.max(game_hist.T[0]) - np.min(game_hist.T[0])
  cluster_height = np.max(game_hist.T[1]) - np.min(game_hist.T[1])

  if cluster_height < 10 and cluster_width < 10:
    # Center the cluster
    # TODO: find better way to center the cluster (currently just pushes it to top left)
    game_hist = game_hist.T
    game_hist[0] -= np.min(game_hist[0])
    game_hist[1] -= np.min(game_hist[1])
    game_hist = game_hist.T

    for i, move in enumerate(game_hist):
      p, v = model.predict(prepare_state(env.board))
      p = p.detach().cpu().numpy()[0]
      v = v[0]
      value_asymmetry += v

      # Add the probability the model would play the human move
      human_metric += p[move[0]][move[1]]
      env.step(move)

    env.reset()

print(human_metric)
print(value_asymmetry)
