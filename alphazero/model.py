import os
import sys
import numpy as np
from copy import deepcopy
from collections import deque

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

sys.path.append('..')

from environment import Environment

from alphazero.mcts import MCTS
from alphazero.database import DataBase
from alphazero.database import prepare_state

torch.manual_seed(80085)
np.random.seed(80085)

def softXEnt (inp, target):
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

  def forward(self, x):
    shortcut = x

    x = self.conv1(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv2(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv3(x)
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

  def forward(self, x):
    shortcut = x

    x = self.conv1(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv2(x)
    x = F.leaky_relu(x, 0.2)

    x = self.conv3(x)

    shortcut = self.conv_change(shortcut)

    x += shortcut
    x = F.leaky_relu(x, 0.2)

    return x

class PolicyHead(nn.Module):
  def __init__(self, board_shape, use_bias):
    super().__init__()
    self.board_shape = board_shape
    self.identity1 = IdentityBlock(3, [24, 48], 24, use_bias)
    self.conv1 = nn.Conv2d(24, 1, padding=(1, 1), kernel_size=3, stride=1, bias=use_bias)
    self.flatten = nn.Flatten()

  def forward(self, x):
    p = self.identity1(x)
    p = self.conv1(p)
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

  def forward(self, x):
    x = self.conv1(x)
    x = F.leaky_relu(x)
    x = self.convolutional1(x)
    x = self.identity1(x)

    p, v = self.policy_head(x), self.value_head(x)
    return p, v

class ZeroTTT():
  def __init__(self, brain_path, opt_path, args={"board_len": 10, "lr": 3e-4, "weight_decay": 1e-4}):
    '''
      brain_path - path to model params
      opt_path - path to optimizer state

      args:
        board_len - # of rows and columns on board
        lr - learning rate
        weight_decay - weight decay
    '''
    self.args = args
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.brain = Brain(input_shape=(2, self.args["board_len"], self.args["board_len"])).to(self.device)

    self.policy_loss = softXEnt
    self.value_loss = nn.MSELoss()
    self.optimizer = optim.AdamW(self.brain.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])

    if brain_path is not None:
      self.load_brain(brain_path, opt_path)

  # TODO: fix for nested Modules
  def get_parameter_count(self):
    return sum(p.numel() for p in self.brain.parameters() if p.requires_grad)

  def save_brain(self, model_name, opt_state_name):
    print("Saving brain...")
    torch.save(self.brain.state_dict(), os.path.join('alphazero/models', model_name))
    if opt_state_name is not None:
        torch.save(self.optimizer.state_dict(), os.path.join('alphazero/models', opt_state_name))

  def load_brain(self, model_name, opt_state_name):
    print("Loading brain...")
    self.brain.load_state_dict(torch.load(os.path.join('alphazero/models', model_name), map_location=self.device))
    if opt_state_name is not None:
        self.optimizer.load_state_dict(torch.load(os.path.join('alphazero/models', opt_state_name), map_location=self.device))
    return

  def predict(self, x, interpret_output=True):
    if len(x.shape) < 4:
      x = np.expand_dims(x, axis=0)

    x = torch.from_numpy(x).float().to(self.device)

    policy, value = self.brain(x)

    if interpret_output: # return 2d policy map and value in usable form
      policy = policy.view(-1, self.args["board_len"], self.args["board_len"])
      policy = policy[0].cpu().detach().numpy()
      value = value[0][0].item()
    return policy, value
