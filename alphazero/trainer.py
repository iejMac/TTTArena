import sys
import torch

sys.path.append(os.path.join(os.environ["HOME"], "AlphaTTT"))

from mcts import MCTS
from database import DataBase
from environment import Environment

class Trainer:
  def __init__(self, model, args):
    '''
      model - trainable class with predict method that returns a valid policy and value
      
      args : 
        mcts_args - dict containing mcts args
        db_args - dict containing database args

        board_len - # of rows and columns on board
    '''

    self.args = args
    self.model = model
    self.database = DataBase(self.args["db_args"])

  def generate_game(self, render=False):
    self.model.brain.eval()
    env = Environment(board_len=self.args["board_len"])

    tau = 1.0
    game_state = 10
    mcts = MCTS(self.model, env.board, self.args["mcts_args"])

    while game_state == 10:
      if len(env.move_hist) > 30: # argmax after 30 moves
        tau = 0.01

      mcts.search()
      self.database.append_policy(((-1)**(env.turn == -1))*env.board, mcts.get_pi())

      move = mcts.select_move(tau=tau)
      game_state = env.step(move)

      if render:
        env.render()

    self.database.append_value(game_state, len(env.move_hist))

    print(f"Player with token: {game_state} won the game in {len(env.move_hist)} moves")

  def train(self, epochs, batch_size, save_id=""):
    print(f"Training on {len(self.database.states)} positions...")
    self.model.brain.train()

    batched_sts, batched_pls, batched_vls = self.database.prepare_batches(batch_size)

    for e in range(epochs):
      for j in range(len(batched_sts)):
        self.model.optimizer.zero_grad()

        batch_st, batch_pl, batch_vl = batched_sts[j], batched_pls[j], batched_vls[j]

        batch_pl = torch.from_numpy(batch_pl).to(self.model.device)
        batch_vl = torch.from_numpy(batch_vl).float().to(self.model.device)
        prob, val = self.model.predict(batch_st, interpret_output=False)
        val = val.flatten()

        p_loss = self.model.policy_loss(prob, batch_pl)
        v_loss = self.model.value_loss(val, batch_vl)

        loss = p_loss + v_loss
        loss.backward()

        self.model.optimizer.step()

    # Save after training step
    self.model.save_brain(f'model_{save_id}', f'opt_state_{save_id}')

  def generate_buffer(self, buffer_path):
    game_nr = 1
    while True:
      print(f"Game {game_nr}...")
      self.generate_game()
      if self.database.is_full():
        self.database.save_data(buffer_path)
        self.database.clear()
      game_nr += 1
