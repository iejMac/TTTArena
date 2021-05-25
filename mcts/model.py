import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from mcts import MCTS
from environment import split_state
from environment import Environment

torch.manual_seed(80085)
np.random.seed(80085)

def softXEnt (inp, target): # temporary
    logprobs = F.log_softmax (inp, dim = 1)
    return  -(target * logprobs).sum() / inp.shape[0]

def append_state(states, labels, state, label):
  # Augmentation
  for i in range(2):
    for j in range(4):
      states.append(np.rot90(state, j))
      labels.append(np.rot90(label, j))
    
    state = state.T
    label = label.T
  
  state = state.T
  label = label.T
  return

class Brain(nn.Module):
  def __init__(self, input_shape=(2, 30, 30)):
    super().__init__()

    self.input_shape = input_shape

    use_bias = True
    self.conv1 = nn.Conv2d(input_shape[0], 24, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv2 = nn.Conv2d(24, 36, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv3 = nn.Conv2d(36, 48, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.conv4 = nn.Conv2d(48, 24, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)

    self.pol_conv1 = nn.Conv2d(24, 12, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)
    self.pol_conv2 = nn.Conv2d(12, 1, padding=(2,2), kernel_size=5, stride=1, bias=use_bias)

    self.val_conv1 = nn.Conv2d(24, 1, kernel_size=3, stride=1, bias=use_bias)
    self.val_linear1 = nn.Linear(64, 50)
    self.val_linear2 = nn.Linear(50, 1)

    self.flatten = nn.Flatten()

  def forward(self, x):
    # Core:
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)

    # Policy Head:
    p = self.pol_conv1(x)
    p = F.relu(p)
    p = self.pol_conv2(p)

    p = p.view(-1, self.input_shape[1]*self.input_shape[2])
    p = F.softmax(p, dim=1)
    p = p.view(-1, self.input_shape[1], self.input_shape[2])

    # Value Head:
    v = self.val_conv1(x)
    v = F.relu(v)
    v = self.flatten(v)
    v = self.val_linear1(v)
    v = F.relu(v)
    v = self.val_linear2(v)
    v = torch.tanh(v)

    return p, v

class ZeroTTT():
  def __init__(self, brain_path=None, opt_path=None, board_len=10, lr=3e-4, weight_decay=0.0):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.brain = Brain(input_shape=(2, board_len, board_len)).to(self.device)
    self.board_len = board_len

    self.optimizer = AdamW(self.brain.parameters(), lr=lr, weight_decay=weight_decay)
    self.value_loss = nn.MSELoss()
    self.policy_loss = nn.CrossEntropyLoss()

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

  def predict(self, x):

    if len(x.shape) < 4:
      x = np.expand_dims(x, axis=0)

    x = torch.from_numpy(x).float().to(self.device)

    policy, value = self.brain(x)
    return policy, value

  def self_play(self, n_games=1, num_simulations=100, positions_per_learn=100, batch_size=20 ,render=10):
    
    # Put model in training mode:
    self.brain.train()
    
    states = []
    policy_labels = []
    value_labels = []
    val_chunk = []

    env = Environment(board_len=self.board_len)

    for game_nr in range(n_games):
      
      mcts = MCTS(self, env.board, num_simulations=num_simulations, alpha=0.25)
      tau = 1.0

      print(f"Game {game_nr+1}...")

      while env.game_over() == 10:

        if len(env.move_hist) > 30: # after 30 moves no randomness
          tau = 0.01

        if np.any(env.board == 0) is False: # tie
          break

        mcts.search()
       
        if env.turn == env.x_token:
          append_state(states, policy_labels, env.board, mcts.get_pi())
        elif env.turn == env.o_token: # swap persepctive so O tokens are positive and X tokens are negative
          append_state(states, policy_labels, (-1)*env.board, mcts.get_pi())

        val_chunk += [env.turn]*8 # accounting for augmentation

        move = mcts.select_move(tau=tau)
        env.step(move)

        if (game_nr+1) % render == 0:
          env.render()

      print(f"Player with token: {env.game_over()} won the game in {len(env.move_hist)} moves")

      if env.game_over() == env.x_token: # pass because the turns correctly specify the return from the proper perspectives
        pass
      elif env.game_over() == env.o_token:
        val_chunk = [lab * (-1.0) for lab in val_chunk] # invert the turns because that will represent -1 return for x turns and 1 for o turns
      else: # tie
        val_chunk = [0 for lab in val_chunk]

      value_labels += val_chunk
      val_chunk = []


      if len(states) >= positions_per_learn: # learn

        print(f"Training on {len(states)} positions...")

        states = [split_state(state) for state in states]

        states = np.array(states)
        policy_labels = np.array(policy_labels)
        value_labels = np.array(value_labels)

        p = np.random.permutation(len(states))

        states = states[p]
        policy_labels = policy_labels[p]
        value_labels = value_labels[p]

        batch_count = int(len(states)/batch_size)
        if len(states) / batch_size > batch_count:
          batch_count += 1

        for j in range(batch_count):

          self.optimizer.zero_grad()

          batch_st = states[j * batch_size: min((j+1) * batch_size, len(states))]
          batch_pl = policy_labels[j * batch_size: min((j+1) * batch_size, len(policy_labels))]
          batch_vl = value_labels[j * batch_size: min((j+1) * batch_size, len(value_labels))]

          batch_pl = torch.from_numpy(batch_pl).to(self.device)
          batch_vl = torch.from_numpy(batch_vl).float().to(self.device)
          prob, val = self.predict(batch_st)
          val = val.flatten()
  
          prob = torch.flatten(prob, 1, 2)
          batch_pl = torch.flatten(batch_pl, 1, 2)
  
          p_loss = softXEnt(prob, batch_pl)
          v_loss = self.value_loss(val, batch_vl)
  
          loss = p_loss + v_loss
          loss.backward()
  
          self.optimizer.step()
  
        states = []
        policy_labels = []
        value_labels = []
        # Save after training step
        self.save_brain('best_model', 'best_opt_state')

      env.reset()
