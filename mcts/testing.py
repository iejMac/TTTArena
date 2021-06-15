import os
import numpy as np

from mcts import MCTS
from model import ZeroTTT
from environment import Environment
from environment import prepare_state

class Test:
  def __init__(self, model_name, opt_name, board_len=10):
    self.board_len = board_len
    self.env = Environment(board_len=board_len)

    self.model = ZeroTTT(brain_path=model_name, opt_path=opt_name)
    self.model.brain.eval()

  def visualize_model_output(self, move_hist, progression=False):
    for move in move_hist:
      if progression:
        p, v = self.model.predict(prepare_state(self.env.board))
        # TODO: make a nicer visualization
        self.env.render()
        print(np.around(p, 3))
        print(v)

      self.env.step(move)

    p, v = self.model.predict(prepare_state(self.env.board))

    self.env.render()
    print(np.around(p, 3))
    print(v)

    self.env.reset()

  def human_game_evaluation(self, data_dir):
    games = os.listdir(data_dir)

    human_move_probability = 0.0
    value_sum = 0.0
    xo_win_moves = [0, 0]

    for game in games:
      game_hist = np.loadtxt(os.path.join(data_dir, game), delimiter=",", dtype=int)

      cluster_width = np.max(game_hist.T[0]) - np.min(game_hist.T[0])
      cluster_height = np.max(game_hist.T[1]) - np.min(game_hist.T[1])

      if cluster_width < self.board_len and cluster_height < self.board_len:
        # Center the cluster:
        game_hist = game_hist.T
        game_hist[0] -= np.min(game_hist[0])
        game_hist[1] -= np.min(game_hist[1])
        game_hist = game_hist.T

        xo_win_moves[len(game_hist) % 2 == 0] += len(game_hist)

        for move in game_hist:
          p, v = self.model.predict(prepare_state(self.env.board))

          # Add the probability the model would play the human move
          human_move_probability += p[move[0]][move[1]]
          value_sum += v

          self.env.step(move)

        self.env.reset()
    print(f"Average human move probability: {human_move_probability/sum(xo_win_moves)}")
    print(f"Average postition evaluation: {value_sum/sum(xo_win_moves)} for winner move distribution: [X, O] = {xo_win_moves}")

  def compare_model(self, opponent_name, opponent_opt_name, games_per_side, num_simulations=100, alpha=0.1, render=10):
    opponent = ZeroTTT(brain_path=opponent_name, opt_path=opponent_opt_name)
    xo_wins = [0, 0]

    for game_nr in range(2*games_per_side):
      print(f"Game {game_nr+1}...")
      mcts_self = MCTS(self.model, self.env.board, num_simulations=num_simulations, alpha=alpha)
      mcts_opponent = MCTS(opponent, self.env.board, num_simulations=num_simulations, alpha=alpha)
      tau = 0.01 # no exploration

      current_player, waiting_player = (mcts_self, mcts_opponent) if game_nr % 2 == 0 else (mcts_opponent, mcts_self)
      game_state = 10

      while game_state == 10:
        current_player.search()
        move = current_player.select_move(tau=tau) # current player selects the move
        waiting_player.select_move(external_move=move) # waiting player updates their mcts to reflect selected move

        game_state = self.env.step(move)
        current_player, waiting_player = waiting_player, current_player
        if (game_nr + 1) % render == 0:
          self.env.render()

      winning_player = "Model" if waiting_player == mcts_self else "Opponent"
      winning_token = 'X' if game_state == 1 else 'O'
      if game_state == 0:
        print(f"It was a tie")
      else:
        print(f"{winning_player} won as {winning_token}")
        if winning_player == "Model":
          xo_wins[winning_token == "O"] += 1

      print(f"Move count: {len(self.env.move_hist)}")
      self.env.reset()

    print(f"Model won {xo_wins[0]}/{games_per_side} games as X ({100*(xo_wins[0]/games_per_side)}%)")      
    print(f"Model won {xo_wins[1]}/{games_per_side} games as O ({100*(xo_wins[1]/games_per_side)}%)")      
    
test = Test("trained_model_8", "trained_opt_state_8", 10)

pos1 = [(5, 5), (4, 5), (4, 4), (3, 6), (4, 6), (3, 5), (2, 6), (3, 7), (2, 7), (3, 4),
(3, 3), (2, 5), (3, 7), (1, 5), (0, 5), (1, 4), (2, 2)]
pos2 = [(0, 0), (5, 5), (5, 0), (5, 4), (0, 9), (5, 3), (7, 1), (5, 6)]
pos3 = [(0, 0), (6, 5), (5, 0), (8, 4), (8, 9), (5, 9), (7, 1), (5, 6)]
pos4 = [(0, 0), (6, 5), (0, 1), (8, 4), (0, 2), (5, 9), (0, 3), (5, 6)]

test.compare_model("trained_model_7", "trained_opt_state_7", games_per_side=20, num_simulations=100, render=10, alpha=0.45)
# test.human_game_evaluation("../data/30x30")
# test.visualize_model_output(pos1, True)
