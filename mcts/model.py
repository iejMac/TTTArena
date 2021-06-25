import os
import numpy as np
from copy import deepcopy
from collections import deque

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from mcts import MCTS
from database import DataBase
from database import prepare_state
from environment import Environment

def softXEnt (inp, target): # temporary
  logprobs = torch.log(inp)
  cross_entropy = -(target * logprobs).sum() / inp.shape[0]
  return cross_entropy

# TODO: try out this variant of residual blocks (diff from paper but same as behavioral_cloning) if doesn't work well
# try the regular BasicBlock (same as paper)

class IdentityBlock(nn.Module):
  def __init__(self, f, filters, input_dim, use_bias=True):
    super().__init__()
    pad = int((f - 1)/2) # same padding
    F1, F2 = filters
    self.conv1 = nn.Conv2d(input_dim, F1, padding=(pad,pad), kernel_size=f, stride=1, bias=use_bias)
    self.conv2 = nn.Conv2d(F1, F2, padding=(pad, pad), kernel_size=f, stride=1, bias=use_bias)
    self.conv3 = nn.Conv2d(F2, F1, padding=(pad, pad), kernel_size=f, stride=1, bias=use_bias)

    # self.bn1 = nn.BatchNorm2d(F1)
    # self.bn2 = nn.BatchNorm2d(F2)
    # self.bn3 = nn.BatchNorm2d(F1)

  def forward(self, x):
    shortcut = x

    x = self.conv1(x)
    # x = self.bn1(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv2(x)
    # x = self.bn2(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv3(x)
    # x = self.bn3(x)
    x += shortcut
    x = F.leaky_relu(x, 0.2)

    return x

class ConvolutionalBlock(nn.Module):
  def __init__(self, f, filters, input_dim, use_bias=True):
    super().__init__()
    pad = int((f - 1)/2) # same padding
    F1, F2, F3 = filters
    self.conv1 = nn.Conv2d(input_dim, F1, padding=(pad, pad), kernel_size=f, stride=1, bias=use_bias)
    self.conv2 = nn.Conv2d(F1, F2, padding=(pad, pad), kernel_size=f, stride=1, bias=use_bias)
    self.conv3 = nn.Conv2d(F2, F3, padding=(pad, pad), kernel_size=f, stride=1, bias=use_bias)
    self.conv_change = nn.Conv2d(input_dim, F3, padding=(0,0), kernel_size=1, stride=1, bias=use_bias)

    # self.bn1 = nn.BatchNorm2d(F1)
    # self.bn2 = nn.BatchNorm2d(F2)
    # self.bn3 = nn.BatchNorm2d(F3)
    # self.bnc = nn.BatchNorm2d(F3)

  def forward(self, x):
    shortcut = x

    x = self.conv1(x)
    # x = self.bn1(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv2(x)
    # x = self.bn2(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv3(x)
    # x = self.bn3(x)

    shortcut = self.conv_change(shortcut)
    # shortcut = self.bnc(shortcut)

    x += shortcut
    x = F.leaky_relu(x, 0.2)

    return x

class PolicyHead(nn.Module):
  def __init__(self, board_shape, use_bias):
    super().__init__()
    self.board_shape = board_shape
    self.identity1 = IdentityBlock(3, [24, 48], 24, use_bias)
    self.conv1 = nn.Conv2d(24, 1, padding=(1, 1), kernel_size=3, stride=1, bias=use_bias)
    # self.bn1 = nn.BatchNorm2d(1)
    self.flatten = nn.Flatten()

  def forward(self, x):
    p = self.identity1(x)
    p = self.conv1(p)
    # p = self.bn1(p)
    p = self.flatten(p)
    p = F.softmax(p, dim=1)
    return p

class ValueHead(nn.Module):
  def __init__(self, use_bias):
    super().__init__()
    self.convolutional1 = ConvolutionalBlock(3, [24, 48, 1], 24, use_bias)
    self.val_linear1 = nn.Linear(100, 1)
    self.flatten = nn.Flatten()

  def forward(self, x):
    v = self.convolutional1(x)
    v = self.flatten(v)
    v = self.val_linear1(v)
    v = torch.tanh(v)
    return v

class Brain(nn.Module):
  def __init__(self, input_shape=(3, 30, 30)):
    super().__init__()

    self.input_shape = input_shape
    use_bias = True
    self.conv1 = nn.Conv2d(input_shape[0], 16, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.convolutional1 = ConvolutionalBlock(5, [24, 48, 24], 16, use_bias)
    self.identity1 = IdentityBlock(5, [24, 48], 24, use_bias)
    self.policy_head = PolicyHead(input_shape, use_bias)
    self.value_head = ValueHead(use_bias)

    # self.bn1 = nn.BatchNorm2d(16)

  def forward(self, x):
    # Core:
    x = self.conv1(x)
    # x = self.bn1(x)
    x = F.leaky_relu(x)
    x = self.convolutional1(x)
    x = self.identity1(x)

    p, v = self.policy_head(x), self.value_head(x)
    return p, v

class ZeroTTT():
  def __init__(self, brain_path=None, opt_path=None, board_len=10, lr=3e-4, weight_decay=0.0):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.brain = Brain(input_shape=(3, board_len, board_len)).to(self.device)
    self.board_len = board_len

    self.optimizer = optim.AdamW(self.brain.parameters(), lr=lr, weight_decay=weight_decay)
    # self.optimizer = optim.SGD(self.brain.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
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
    
    database = DataBase(max_len=max_position_storage)

    positions_to_next_learn = positions_per_learn

    env = Environment(board_len=self.board_len)

    for game_nr in range(n_games):
      
      self.brain.eval()
      mcts = MCTS(self, env.board, alpha=0.35)
      tau = 1.0

      print(f"Game {game_nr+1}...")
      game_state = 10

      while game_state == 10:

        if len(env.move_hist) > 30: # after 30 moves no randomness
          tau = 0.01

        mcts.search(num_simulations=num_simulations)
        database.append_policy(env.board, mcts.get_pi(), augmentations=["flip", "rotate"])

        move = mcts.select_move(tau=tau)
        game_state = env.step(move)

        if (game_nr+1) % render == 0:
          env.render()

      database.append_policy(env.board, mcts.get_pi(), augmentations=["flip", "rotate"]) # append terminal state
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
