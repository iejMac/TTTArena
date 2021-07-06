import os
import numpy as np

from mcts import MCTS
from model import ZeroTTT
from database import prepare_state
from environment import Environment

class Test:
  def __init__(self, model_name, opt_name, board_len=10):
    self.board_len = board_len
    self.env = Environment(board_len=board_len)

    self.model = ZeroTTT(brain_path=model_name, opt_path=opt_name)
    self.model.brain.eval()

  def visualize_model_output(self, move_hist, progression=False):
    for move in move_hist:
      if progression:
        p, v = self.model.predict(prepare_state(((-1)**(self.env.turn == -1))*self.env.board))
        # TODO: make a nicer visualization
        self.env.render()
        print(np.around(p, 3))
        print(v)

      self.env.step(move)

    p, v = self.model.predict(prepare_state(((-1)**(self.env.turn == -1))*self.env.board))
    self.env.render()
    print(np.around(p, 3))
    print(v)
    print(f"{'X' if self.env.turn == 1 else 'O'}'s turn to move") 

    self.env.reset()

  def human_game_evaluation(self, data_dir):
    games = os.listdir(data_dir)

    human_move_probability = 0.0
    value_sum = 0.0
    wl_certainty = [0.0, 0.0]
    wl_certainty_count = [0, 0]
    value_mse = 0.0
    correct_terminal_state_sign = 0
    xo_win_moves = [0, 0]
    compatible_games = 0

    for game in games:
      game_hist = np.loadtxt(os.path.join(data_dir, game), delimiter=",", dtype=int)

      cluster_width = np.max(game_hist.T[0]) - np.min(game_hist.T[0])
      cluster_height = np.max(game_hist.T[1]) - np.min(game_hist.T[1])

      if cluster_width < self.board_len and cluster_height < self.board_len:
        compatible_games += 1
        # Center the cluster:
        game_hist = game_hist.T
        game_hist[0] -= np.min(game_hist[0])
        game_hist[1] -= np.min(game_hist[1])
        game_hist = game_hist.T

        xo_win_moves[len(game_hist) % 2 == 0] += len(game_hist) + 1 # adding terminal state evaluation
        winner = 1 if len(game_hist) % 2 != 0 else -1

        for move in game_hist:
          p, v = self.model.predict(prepare_state(((-1)**(self.env.turn == -1))*self.env.board))

          # Add the probability the model would play the human move
          human_move_probability += p[move[0]][move[1]]
          value_sum += v
          wl_certainty[v < 0.0] += abs(v)
          wl_certainty_count[v < 0.0] += 1
          value_mse += (winner*((-1)**(self.env.turn == -1)) - v)**2

          self.env.step(move)

        p, v = self.model.predict(prepare_state(((-1)**(self.env.turn == -1))*self.env.board)) # check evaluation on terminal state
        value_sum += v
        value_mse += (winner*((-1)**(self.env.turn == -1)) - v)**2
        correct_terminal_state_sign += (v < 0)

        self.env.reset()
    print(f"Average human move probability: {human_move_probability/sum(xo_win_moves)}")
    print(f"Average position evaluation MSE: {value_mse/sum(xo_win_moves)}")
    print(f"Average evaluation certainty: [W, L] = {wl_certainty[0]/(wl_certainty_count[0]+0.1)}, {wl_certainty[1]/(wl_certainty_count[1]+0.1)}")
    print(f"Average postition evaluation: {value_sum/sum(xo_win_moves)} for winning position distribution: [X, O] = {xo_win_moves}")
    print(f"Evaluated {correct_terminal_state_sign}/{compatible_games} terminal states with correct sign")

  def compare_model(self, opponent_name, opponent_opt_name, games_per_side, num_simulations=100, alpha=0.1, render=10):
    opponent = ZeroTTT(brain_path=opponent_name, opt_path=opponent_opt_name)
    xo_wins = [0, 0]

    for game_nr in range(2*games_per_side):
      print(f"Game {game_nr+1}...")
      mcts_self = MCTS(self.model, self.env.board, alpha=alpha)
      mcts_opponent = MCTS(opponent, self.env.board, alpha=alpha)
      tau = 0.01 # no exploration

      current_player, waiting_player = (mcts_self, mcts_opponent) if game_nr % 2 == 0 else (mcts_opponent, mcts_self)
      game_state = 10

      while game_state == 10:
        current_player.search(num_simulations)
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

  def play_model(self, player="X", num_simulations=100, alpha=0.1):
    mcts = MCTS(self.model, self.env.board, alpha=alpha)
    game_state = 10
    tau = 0.01
    turn = 1 if player == "X" else -1 # your move vs model move

    self.env.render()

    while game_state == 10:
      if turn == 1:
        move_str = input("Input a move y, x: ")
        move = tuple(int(x) for x in move_str.split(","))
        mcts.select_move(external_move=move)
      else:
        mcts.search(num_simulations=num_simulations)
        move = mcts.select_move(tau=tau)

      game_state = self.env.step(move)
      turn *= -1

      self.env.render()

    # TODO: hack assumes, no ties
    winning_token = "X" if game_state == 1 else "O"
    winner = "You" if winning_token == player else "Model"
    print(f"{winner} won in {len(self.env.move_hist)} moves")
    self.env.reset()
    
test = Test("trained_model_4_1", "trained_opt_state_4_1", 10) 

pos1 = [(5, 5), (4, 5), (4, 4), (3, 6), (4, 6), (3, 5), (2, 6), (3, 7), (2, 7), (3, 4),
(3, 3), (2, 5), (3, 8), (1, 5), (0, 5), (1, 4), (2, 2)]
pos2 = [(0, 0), (5, 5), (5, 0), (5, 4), (0, 9), (5, 3), (7, 1), (5, 6)]
pos3 = [(0, 0), (6, 5), (5, 0), (8, 4), (8, 9), (5, 9), (7, 1), (5, 6)]
pos4 = [(0, 0), (6, 5), (0, 1), (8, 4), (0, 2), (5, 9), (0, 3), (5, 6)]
pos5 = [(3, 4), (5, 5), (3, 5), (5, 4), (3, 6), (4, 4), (3, 7), (4, 5), (3, 8)]

# test.play_model(player="X", num_simulations=50)
# test.play_model(player="X", num_simulations=800)
test.compare_model("trained_model_3", "trained_opt_state_3", games_per_side=40, num_simulations=100, render=100, alpha=0.1)
# test.human_game_evaluation("../data/30x30")
# test.visualize_model_output(pos1, False)
# test.visualize_model_output(pos2, False)
# test.visualize_model_output(pos3, False)
# test.visualize_model_output(pos4, False)
# test.visualize_model_output(pos5, False)
