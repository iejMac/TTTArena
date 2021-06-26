import os
import numpy as np
from copy import deepcopy

from database import DataBase
from environment import Environment

data_dir = "../data/30x30"
board_len = 10

games = os.listdir(data_dir)
env = Environment(board_len)
db = DataBase(max_len=1000000)

for game in games:
  game_hist = np.loadtxt(os.path.join(data_dir, game), delimiter=",", dtype=int)

  cluster_height = np.max(game_hist.T[0]) - np.min(game_hist.T[0])
  cluster_width = np.max(game_hist.T[1]) - np.min(game_hist.T[1])

  if cluster_width < board_len and cluster_height < board_len:

    # Set cluster to top-left
    game_hist = game_hist.T
    game_hist[0] -= np.min(game_hist[0])
    game_hist[1] -= np.min(game_hist[1])
    game_hist = game_hist.T

    y_free = board_len - cluster_height
    x_free = board_len - cluster_width

    winner = (-1.0)**(len(game_hist) % 2 == 0)

    for dy in range(y_free):
      for dx in range(x_free):
         
        hist = deepcopy(game_hist).T
        hist[0] += dy
        hist[1] += dx
        hist = hist.T

        for i, move in enumerate(hist):
          policy = np.zeros((board_len, board_len))
          policy[move[0]][move[1]] = 1.0
          
          db.append_policy(deepcopy(env.board), policy, ["flip", "rotate"])
          env.step(move)

        db.append_value(winner, len(game_hist) - 1) # no terminal state
        env.reset()

# db.save_data()
