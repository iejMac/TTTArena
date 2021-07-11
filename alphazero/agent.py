import os
import sys
sys.path.append(os.path.join(os.environ["HOME"], "AlphaTTT"))

from agent import Agent

from alphazero.mcts import MCTS
from alphazero.model import ZeroTTT

default_model_args = {
  "board_len": 10,
  "lr": 3e-4,
  "weight_decay": 1e-4
}

default_mcts_args = {
  "num_simulations": 1000,
  "alpha": 0.1,
  "c_puct": 4,
  "dirichlet_alpha": 0.3,
  "tau": 0.01
}

class ZeroAgent(Agent):
  def __init__(self, model_name, opt_state_name, args):
    super().__init__(model_name)
    self.model = ZeroTTT(model_name, opt_state_name, args["model_args"])
    self.model.brain.eval()
    self.args = args

  def init_state(self, state):
    self.mcts = MCTS(self.model, state, self.args["mcts_args"])

  def make_action(self, state):
    self.mcts.search()
    move = self.mcts.select_move(tau=self.args["mcts_args"]["tau"])
    return move

  def update_state(self, action):
    self.mcts.select_move(external_move=action)

  @staticmethod
  def get_params():
    model_name = input("Model name: ")
    opt_state_name = input("Optimizer state name: ")

    if input("Would you like to adjust MCTS args? ") != "":
      for key in default_mcts_args:
        default_mcts_args[key] = type(default_mcts_args[key])(input(f"{key}: "))
        
    args = {
      "model_args": default_model_args,
      "mcts_args": default_mcts_args
    }

    return (model_name, opt_state_name, args)
