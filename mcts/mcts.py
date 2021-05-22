import math
import random
import numpy as np

from environment import split_state

np.random.seed(80085)
random.seed(80085)

def ucb_score(child_prior, parent_visit_count, child_visit_count):
  # pb_c = math.log((parent_visit_count + 19652 + 1)/19652) + 1.25
  pb_c = math.sqrt(parent_visit_count) / (child_visit_count + 1)
  prior_score = pb_c * child_prior
  return prior_score

def np_softmax(arr_2d, dim=2):

  if dim == 2:
    flat = arr_2d.flatten()
  elif dim == 1:
    flat = arr_2d

  e_x = np.exp(flat - np.max(flat))
  e_x = e_x / e_x.sum()
  e_x = e_x.reshape(arr_2d.shape)
  return e_x

class Node():
  def __init__(self, state, dir_alpha=0.25):
    self.state = state
    self.children = []
    self.visit_count = 0
    self.alpha = dir_alpha

  def is_leaf_node(self):
    return (len(self.children) == 0)

  def expand(self, model, is_root=False):

    if self.is_leaf_node() is False: # safety so a node doesn't double its children
      return

    p_vals, value = model.predict(split_state(self.state))
    p_vals = p_vals[0].cpu().detach().numpy()

    # Add dirichlet noise if root state:
    if is_root is True:
      dirichlet = np.random.dirichlet([0.03]*len(self.state)*len(self.state)).reshape((self.state.shape))
      p_vals = (1 - self.alpha) * p_vals + self.alpha * dirichlet

    for i, row in enumerate(p_vals):
      for j, prior_prob in enumerate(row):
        if self.state[i][j] == 0: # only append available moves
          self.children.append(Edge(prior_prob, (i, j)))

    # Quick experiment:
    random.shuffle(self.children)

    return (-1.0)*value # negative value because this is evaluated from the position of the opposite player

  def find_leaf(self, model):

    self.visit_count += 1

    if self.is_leaf_node():
      return self.expand(model)

    # find child node that maximuzes Q + U
    max_ind, max_val = 0, 0.0
    for i, child_node in enumerate(self.children):
      val = child_node.Q + ucb_score(child_node.P, self.visit_count, child_node.N)
      if val > max_val:
        max_ind = i
        max_val = val

    v = self.children[int(max_ind)].traverse(self.state, model)
    return (-1.0)*v

class Edge():
  def __init__(self, p, action):

    self.action = action
    self.N = 0
    self.P = p
    self.W = 0
    self.Q = 0

    self.node = None

  def initialize_node(self, state): # destination node doesn't need to be initialized all the time, only if we're actually going to use it
    next_state = np.copy(state)
    next_state[self.action[0]][self.action[1]] = 1
    self.node = Node(next_state * (-1)) # multiply state by -1 to swap to opposite perspective
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
  def __init__(self, model, root_state, num_simulations=1600):
    self.model = model
    self.root = Node(root_state)
    self.root.expand(self.model, is_root=True)
    self.num_simulations = num_simulations # represents "thinking time"

  def search(self): # builds the search tree from the root node
    for i in range(self.num_simulations):
      self.root.find_leaf(self.model)

    return

  def select_move(self, tau=0.0, external_move=None):

    if external_move is None:
      probas = []
      for i, child in enumerate(self.root.children):
        probas.append(child.N)
      
      # probas = probas / np.sum(probas)
      probas = np_softmax(np.array(probas), dim=1)

      if tau == 0.0:
        max_ind = np.argmax(probas)
      else:
        max_ind = int(np.random.choice(len(probas), 1, p=probas))

    else:
      max_ind = 0
      for i, child in enumerate(self.root.children):
        if child.action == external_move:
          max_ind = i
          break

    # need to update the search tree 
    chosen = self.root.children[max_ind]
    if chosen.node == None:
      chosen.initialize_node(self.root.state)
    self.root = chosen.node
    return chosen.action

  def get_pi(self, tau=1.0, as_prob=True):
    move_dist = np.zeros((len(self.root.state), len(self.root.state)))

    for child in self.root.children:
      move_dist[child.action[0]][child.action[1]] = child.N

    if as_prob is True:
      move_dist = np_softmax(move_dist)

    return move_dist
