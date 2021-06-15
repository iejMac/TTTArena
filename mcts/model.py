import os
import numpy as np
from copy import deepcopy
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from mcts import MCTS
from database import DataBase
from environment import Environment
from environment import prepare_state

def softXEnt (inp, target): # temporary
  logprobs = torch.log(inp)
  cross_entropy = -(target * logprobs).sum() / inp.shape[0]
  return cross_entropy

class PolicyHead(nn.Module):
  def __init__(self, board_shape, use_bias):
    super().__init__()

    self.board_shape = board_shape

    self.pol_conv1 = nn.Conv2d(32, 32, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.pol_conv2 = nn.Conv2d(32, 12, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.pol_conv3 = nn.Conv2d(12, 2, padding=(2,2), kernel_size=3, stride=1, bias=use_bias)

    self.pol_linear1 = nn.Linear(288, board_shape[1]*board_shape[2])

    self.bn1 = nn.BatchNorm2d(32)
    self.bn2 = nn.BatchNorm2d(12)
    self.bn3 = nn.BatchNorm2d(2)

    self.flatten = nn.Flatten()

  def forward(self, x):
    p = self.pol_conv1(x)
    # p = self.bn1(p)
    p = F.leaky_relu(p, 0.2)
    p = self.pol_conv2(p)
    # p = self.bn2(p)
    p = F.leaky_relu(p, 0.2)
    p = self.pol_conv3(p)
    # p = self.bn3(p)
    p = F.leaky_relu(p, 0.2)

    p = self.flatten(p)

    p = self.pol_linear1(p)
    p = F.softmax(p, dim=1)
    return p

class ValueHead(nn.Module):
  def __init__(self, use_bias):
    super().__init__()
    self.val_conv1 = nn.Conv2d(32, 24, kernel_size=5, stride=1, bias=use_bias)
    self.val_conv2 = nn.Conv2d(24, 4, kernel_size=3, stride=1, bias=use_bias)

    self.val_linear1 = nn.Linear(64, 50)
    self.val_linear2 = nn.Linear(50, 1)

    dropout_prob = 0.5
    self.dp1 = nn.Dropout2d(dropout_prob)
    self.dp2 = nn.Dropout2d(dropout_prob)
    self.dp3 = nn.Dropout(dropout_prob)

    # self.bn1 = nn.BatchNorm2d(24)
    # self.bn2 = nn.BatchNorm2d(4)

    self.flatten = nn.Flatten()

  def forward(self, x):
    v = self.val_conv1(x)
    v = F.leaky_relu(v, 0.2)
    v = self.val_conv2(v)
    v = F.leaky_relu(v, 0.2)

    v = self.flatten(v)
    v = self.val_linear1(v)
    v = F.leaky_relu(v, 0.2)
    v = self.val_linear2(v)
    v = torch.tanh(v)
    return v

class Brain(nn.Module):
  def __init__(self, input_shape=(3, 30, 30)):
    super().__init__()

    self.input_shape = input_shape

    use_bias = True
    dropout_prob = 0.5

    self.conv1 = nn.Conv2d(input_shape[0], 64, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv2 = nn.Conv2d(64, 64, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv3 = nn.Conv2d(64, 48, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv4 = nn.Conv2d(48, 32, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)

    self.dp1 = nn.Dropout2d(dropout_prob)
    self.dp2 = nn.Dropout2d(dropout_prob)
    self.dp3 = nn.Dropout2d(dropout_prob)
    self.dp4 = nn.Dropout2d(dropout_prob)

    # self.bn1 = nn.BatchNorm2d(64)
    # self.bn2 = nn.BatchNorm2d(96)
    # self.bn3 = nn.BatchNorm2d(96)
    # self.bn4 = nn.BatchNorm2d(48)

    self.policy_head = PolicyHead(input_shape, use_bias)
    self.value_head = ValueHead(use_bias)

  def forward(self, x):
    # Core:
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv3(x)
    x = F.leaky_relu(x, 0.2)
    x = self.conv4(x)
    x = F.leaky_relu(x, 0.2)

    p, v = self.policy_head(x), self.value_head(x)

    return p, v

class ZeroTTT():
  def __init__(self, brain_path=None, opt_path=None, board_len=10, lr=3e-4, weight_decay=0.0):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.brain = Brain(input_shape=(3, board_len, board_len)).to(self.device)
    self.board_len = board_len

    self.optimizer = AdamW(self.brain.parameters(), lr=lr, weight_decay=weight_decay)
    self.value_loss = nn.MSELoss()
    self.policy_loss = softXEnt

    if brain_path is not None:
      self.load_brain(brain_path, opt_path)

  def get_parameter_count(self):
    return sum(p.numel() for p in self.brain.parameters() if p.requires_grad)

  def save_brain(self, model_name, opt_state_name):
    print("Saving brain...")
    torch.save(self.brain.state_dict(), os.path.join('models', model_name))
    if opt_state_name is not None:
        torch.save(self.optimizer.state_dict(), os.path.join('models', opt_state_name))

  def load_brain(self, model_name, opt_state_name):
    print("Loading brain...")
    self.brain.load_state_dict(torch.load(os.path.join('models', model_name), map_location=self.device))
    if opt_state_name is not None:
        self.optimizer.load_state_dict(torch.load(os.path.join('models', opt_state_name), map_location=self.device))
    return

  def predict(self, x, interpret_output=True):

    if len(x.shape) < 4:
      x = np.expand_dims(x, axis=0)

    x = torch.from_numpy(x).float().to(self.device)

    policy, value = self.brain(x)

    if interpret_output: # return 2d policy map and value in usable form
      policy = policy.view(-1, self.board_len, self.board_len)
      policy = policy[0].cpu().detach().numpy()
      value = value[0][0].item()

    return policy, value

  def self_play(self, n_games=1, num_simulations=100, training_epochs=1, positions_per_learn=100, max_position_storage=100, batch_size=20, render=10, generate_buffer_path=None):
    '''
    num_simulations : limit leaf expansions for monte-carlo rollouts

    max_position_storage : maximum amount of positions stored in RAM

    positions_per_learn : train model every time this many positions are added to buffer

    generate_buffer_path : if passed a path in string form, self-play will just generate games and
    save them to a replay_buffer at the given path to train on later in separate algorithm.
    '''
    
    database = DataBase(max_len=max_position_storage, augmentations=["flip", "rotate"])

    positions_to_next_learn = positions_per_learn

    env = Environment(board_len=self.board_len)

    for game_nr in range(n_games):
      
      self.brain.eval()
      mcts = MCTS(self, env.board, alpha=0.75)
      tau = 1.0

      print(f"Game {game_nr+1}...")
      game_state = 10

      while game_state == 10:

        if len(env.move_hist) > 30: # after 30 moves no randomness
          tau = 0.01

        mcts.search(num_simulations=num_simulations)
        database.append_policy(env.board, mcts.get_pi())

        move = mcts.select_move(tau=tau)
        game_state = env.step(move)

        if (game_nr+1) % render == 0:
          env.render()

      database.append_policy(env.board, mcts.get_pi()) # append terminal state
      print(f"Player with token: {game_state} won the game in {len(env.move_hist)} moves")

      database.append_value(game_state, len(env.move_hist))
      positions_to_next_learn -= (len(env.move_hist)+1)*database.augmentation_coefficient

      if database.is_full() and generate_buffer_path is not None:
        database.save_data(generate_buffer_path)
        database.clear()
      elif database.is_full() and positions_to_next_learn <= 0: # learn
        self.brain.train()
        print(f"Training on {len(database.states)} positions...")

        batched_sts, batched_pls, batched_vls = database.prepare_batches(batch_size)

        for e in range(training_epochs):
          for j in range(len(batched_sts)):
            self.optimizer.zero_grad()

            batch_st, batch_pl, batch_vl = batched_sts[j], batched_pls[j], batched_vls[j]

            batch_pl = torch.from_numpy(batch_pl).to(self.device)
            batch_vl = torch.from_numpy(batch_vl).float().to(self.device)
            prob, val = self.predict(batch_st, interpret_output=False)
            val = val.flatten()

            p_loss = self.policy_loss(prob, batch_pl)
            v_loss = self.value_loss(val, batch_vl)

            loss = p_loss + v_loss
            loss.backward()
    
            self.optimizer.step()
  
        # Save after training step
        self.save_brain('best_model', 'best_opt_state')
        positions_to_next_learn = positions_per_learn

      env.reset()
