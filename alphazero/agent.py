import sys
sys.path.append('..')

from agent import Agent

from alphazero.mcts import MCTS
from alphazero.model import ZeroTTT

class ZeroAgent(Agent):
  def __init__(self, model_name, opt_state_name, args):
    super().__init__(model_name)
    self.model = ZeroTTT(model_name, opt_state_name, args["model_args"])
    self.args = args

  def init_state(self, state):
    self.mcts = MCTS(self.model, state, self.args["mcts_args"])

  def make_action(self, state):
    self.mcts.search()
    move = self.mcts.select_move(tau=self.args["mcts_args"]["tau"])
    return move

  def update_state(self, action):
    self.mcts.select_move(external_move=action)

