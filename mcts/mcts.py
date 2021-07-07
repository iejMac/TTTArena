import math
import random
import numpy as np
from copy import deepcopy

from database import prepare_state
from environment import Environment

np.random.seed(80085)
random.seed(80085)

def PUCT_score(child_value, child_prior, parent_visit_count, child_visit_count, c_puct):
  pb_c = child_prior * math.sqrt(parent_visit_count) / (child_visit_count + 1)
  return child_value + c_puct * pb_c

class MCTS():
  def __init__(self, model, root_state, args):
    '''
      model - class with predict method that returns a valid policy and value
      root_state - board_len x board_len array with the initial state of the game

      args:
        num_simulations - number of leaf node expansions per search
        alpha - mixing constant between policy and dirichlet noise
        dirichlet_alpha - dirichlet constant for generating dirichlet distribution
        c_puct - exploration constant in PUCT score
    '''

    self.model = model
    self.root = deepcopy(root_state)
    self.args = args

    self.Qsa = {} # self.Qsa(s, a) = Q value for (s, a)
    self.Nsa = {} # self.Nsa(s, a) = (s, a) visit count
    self.Ns = {} # self.Ns(s) = s visit count
    self.Ps = {} # self.Ps(s) = list of available actions in s and corresponding raw probabilities

    self.Es = {} # terminal states, potentially going to do this if not too computationally expensive and dirty

    # Add dirichlet noise to initial root node
    self.add_dirichlet()

  def add_dirichlet(self):
    rs = self.root.tobytes()
    if rs not in self.Ps:
      self.find_leaf(deepcopy(self.root))
    if self.Es[rs] == 10:
      dirichlet = np.random.dirichlet([self.args["dirichlet_alpha"]]*len(self.Ps[rs]))
      for i, (move, prob) in enumerate(self.Ps[rs]):
        self.Ps[rs][i] = (move, (1 - self.args["alpha"]) * prob + dirichlet[i] * self.args["alpha"])

  def search(self): # builds the search tree from the root node
    for i in range(self.args["num_simulations"]):
      self.find_leaf(deepcopy(self.root))
    return

  def find_leaf(self, state):
    s = state.tobytes()

    if s not in self.Es:
      self.Es[s] = Environment.game_over(state)
    if self.Es[s] != 10:
      # terminal state
      return -self.Es[s]

    if s not in self.Ps: # expand leaf node
      p, v = self.model.predict(prepare_state(state)) 
      availability_mask = (state == 0)
      p *= availability_mask
      if np.sum(p) > 0.0:
        p /= np.sum(p) # re-normalize

      move_probs = []

      for i, row in enumerate(p): 
        for j, prob in enumerate(row):
          if state[i][j] == 0:
            move_probs.append(((i, j), prob))
   
      self.Ps[s] = move_probs
      self.Ns[s] = 1
      return -v

    max_puct = -float('inf')
    max_action = None

    for move, prob in self.Ps[s]:
      (Nc, Qc) = (self.Nsa[(s, move)], self.Qsa[(s, move)]) if (s, move) in self.Nsa else (0, 0.0)
      puct = PUCT_score(Qc, prob, self.Ns[s], Nc, self.args["c_puct"])
      if puct > max_puct:
        max_puct = puct
        max_action = move

    a = max_action
    state[a] = 1
    state *= -1

    v = self.find_leaf(state)

    if (s, a) in self.Nsa:
      self.Nsa[(s, a)] += 1
      self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
    else:
      self.Nsa[(s, a)] = 1
      self.Qsa[(s, a)] = v
      
    self.Ns[s] += 1
    return -v

  def get_pi(self, tau=1.0, as_prob=True):
    move_dist = np.zeros((len(self.root), len(self.root)))
    rs = self.root.tobytes()
    for move, _ in self.Ps[rs]:
      move_dist[move] = self.Nsa[(rs, move)] if (rs, move) in self.Nsa else 0
    if as_prob is True:
      move_dist = np.power(move_dist, 1.0/tau)
      if np.sum(move_dist) > 0.0:
        move_dist /= np.sum(move_dist)
    return move_dist

  def select_move(self, tau=1.0, external_move=None):
    if external_move is None:
      probas = self.get_pi(tau)
      selected_move = int(np.random.choice(len(probas.flatten()), 1, p=probas.flatten()))
      selected_move = np.unravel_index(selected_move, probas.shape)
    else:
      selected_move = external_move

    self.root[selected_move] = 1
    self.root *= -1

    # Add dirichlet noise to new root node:
    self.add_dirichlet()

    return selected_move
