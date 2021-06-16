import math
import random
import numpy as np
from copy import deepcopy

from environment import prepare_state

def PUCT_score(child_value, child_prior, parent_visit_count, child_visit_count):
  c_puct = 8
  pb_c = child_prior * math.sqrt(parent_visit_count) / (child_visit_count + 1)
  puct = child_value + c_puct * pb_c
  return puct

class Node():
  def __init__(self, state):
    self.state = state
    self.children = []
    self.visit_count = 0

  def is_leaf_node(self):
    return (len(self.children) == 0)

  def expand(self, model):

    if self.is_leaf_node() is False: # safety so a node doesn't double its children
      return

    p_vals, value = model.predict(prepare_state(self.state))

    for i, row in enumerate(p_vals):
      for j, prior_prob in enumerate(row):
        if self.state[i][j] != 0: # zero out probabilities for unavailable moves
          p_vals[i][j] = 0.0
    if np.sum(p_vals) > 0.0:
      p_vals = p_vals / np.sum(p_vals) # re-normalize

    for i, row in enumerate(p_vals):
      for j, prior_prob in enumerate(row):
        if self.state[i][j] == 0:
          self.children.append(Edge(prior_prob, (i, j)))

    # Quick experiment:
    random.shuffle(self.children)

    reverse_value = np.sum(self.state) == 0 # this means this state is after an O move
    return ((-1.0)**reverse_value)*value # negative value because this is evaluated from the position of the opposite player
    # return value

  def find_leaf(self, model):

    self.visit_count += 1

    if self.is_leaf_node():
      return self.expand(model)

    # find child node that maximuzes Q + U
    max_ind, max_val = 0, PUCT_score(self.children[0].Q, self.children[0].P, self.visit_count, self.children[0].N)
    for i, child_node in enumerate(self.children):
      if self.state[child_node.action[0]][child_node.action[1]] == 0:
        val = PUCT_score(child_node.Q, child_node.P, self.visit_count, child_node.N)
      else:
        continue
      if val > max_val:
        max_ind = i
        max_val = val

    v = self.children[int(max_ind)].traverse(self.state, model)
    return (-1.0)*v
    # return v

class Edge():
  def __init__(self, p, action):

    self.action = action
    self.N = 0
    self.P = p
    self.W = 0
    self.Q = 0

    self.node = None

  def initialize_node(self, state): # destination node doesn't need to be initialized all the time, only if we're actually going to use it
    next_state = deepcopy(state)
    turn = (-1)**(np.sum(next_state) > 0)
    next_state[self.action[0]][self.action[1]] = turn
    # self.node = Node(next_state * (-1)) # multiply state by -1 to swap to opposite perspective
    self.node = Node(next_state)
    return

  def traverse(self, state, model):
    if self.node is None:
      self.initialize_node(state)

    v =  self.node.find_leaf(model)

    self.N += 1
    self.W += v
    self.Q = self.W / self.N

    return v


class MCTS():
  def __init__(self, model, root_state, alpha=0.25):
    self.model = model
    self.root = Node(root_state)
    self.root.expand(self.model)
    self.alpha = alpha
    # Add dirichlet noise to root node:
    dirichlet = np.random.dirichlet([0.3]*len(self.root.children))
    for i, child in enumerate(self.root.children):
      if self.root.state[child.action[0]][child.action[1]] == 0.0:
        child.P = (1 - self.alpha)*child.P + dirichlet[i] * self.alpha

  def search(self, num_simulations=200): # builds the search tree from the root node
    for i in range(num_simulations):
      self.root.find_leaf(self.model)
    return

  def get_pi(self, tau=1.0, as_prob=True):
    move_dist = np.zeros((len(self.root.state), len(self.root.state)))
    for child in self.root.children:
      move_dist[child.action[0]][child.action[1]] = child.N
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

    max_ind = 0
    for i, child in enumerate(self.root.children):
      if child.action == selected_move:
        max_ind = i
        break

    # need to update the search tree 
    chosen = self.root.children[max_ind]
    if chosen.node == None:
      chosen.initialize_node(self.root.state)
    self.root = chosen.node

    # Add dirichlet noise to new root node:
    dirichlet = np.random.dirichlet([0.3]*len(self.root.children))
    for i, child in enumerate(self.root.children):
      if self.root.state[child.action[0]][child.action[1]] == 0.0:
        child.P = (1 - self.alpha)*child.P + dirichlet[i] * self.alpha

    child_sum = sum([c.P for c in self.root.children]) # re-normalize because we don't add dirichlet to all children
    for child in self.root.children:
      child.P /= child_sum

    return chosen.action
