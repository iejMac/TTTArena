import os
import numpy as np
from copy import deepcopy
from collections import deque

from environment import prepare_state

def rotate_augmentation(states, labels):
  aug_states, aug_labels = [], []
  for i in range(len(states)):
    for j in range(4):
      aug_states.append(deepcopy(np.rot90(states[i], j)))
      aug_labels.append(deepcopy(np.rot90(labels[i], j)))
  return aug_states, aug_labels
    
def flip_augmentation(states, labels):
  aug_states, aug_labels = [], []
  for i in range(len(states)):
    aug_states += [deepcopy(states[i]), deepcopy(states[i].T)]
    aug_labels += [deepcopy(labels[i]), deepcopy(labels[i].T)]
  return aug_states, aug_labels

class DataBase:
  def __init__(self, max_len=600, augmentations=[]):
    '''
    augmentations : 
      flip - transpose the state and policy (2x)
      rotate - rotate state and policy by 90 degrees (4x)

    max_len :
      maximum length of replay buffer in RAM
    '''

    self.max_len = max_len
    self.augmentation_coefficient = 1
    self.augmentations = augmentations

    self.states = deque([], maxlen=max_len)
    self.policy_labels = deque([], maxlen=max_len)
    self.value_labels = deque([], maxlen=max_len)

  def clear(self):
    self.states.clear()
    self.policy_labels.clear()
    self.value_labels.clear()

  def is_full(self):
    return len(self.states) == len(self.policy_labels) == len(self.value_labels) == self.max_len

  def append_policy(self, state, policy_label):
    aug_states = [state]
    aug_policy_labels = [policy_label]

    for aug in self.augmentations:
      aug_states, aug_policy_labels = eval(aug + "_augmentation")(aug_states, aug_policy_labels)

    # Temporary:
    aug_policy_labels = [aug.flatten() for aug in aug_policy_labels]

    self.states += aug_states 
    self.policy_labels += aug_policy_labels
    self.augmentation_coefficient = len(aug_states)

  def append_value(self, winner, game_length):
    self.value_labels += [winner]*((game_length + 1)*self.augmentation_coefficient)

  def save_data(self, path="./replay_buffer"):
    prepared_states = [prepare_state(state) for state in self.states]
    states = np.array(prepared_states)
    pol_labels = np.array(self.policy_labels)
    val_labels = np.array(self.value_labels)

    try:
      largest_index = max([int(name[12:-4]) for name in os.listdir(os.path.join(path, "states"))])
    except ValueError:
      largest_index = -1

    print(f"Saving replay chunk #{largest_index+1} of {len(states)} positions...")
    np.save(os.path.join(path, "states", f"state_chunk_{largest_index+1}"), states)
    np.save(os.path.join(path, "policy_labels", f"policy_chunk_{largest_index+1}"), pol_labels)
    np.save(os.path.join(path, "value_labels", f"value_chunk_{largest_index+1}"), val_labels)

  def prepare_batches(self, batch_size, from_memory_paths=None):

    assert(len(self.states) == len(self.policy_labels) == len(self.value_labels))

    # Numpy-ify
    if from_memory_paths is None:
      train_states = np.array([prepare_state(state) for state in self.states])
      train_policy_labels = np.array(self.policy_labels)
      train_value_labels = np.array(self.value_labels)
    else:
      train_states = np.load(os.path.join(from_memory_paths[0]))
      train_policy_labels = np.load(os.path.join(from_memory_paths[1]))
      train_value_labels = np.load(os.path.join(from_memory_paths[2]))

    # Mix up:
    perm = np.random.permutation(len(train_states))
    train_states = train_states[perm]
    train_policy_labels = train_policy_labels[perm]
    train_value_labels = train_value_labels[perm]

    # Cut off incomplete batch
    batch_count = int(len(train_states)/batch_size)
    train_states = train_states[:batch_count*batch_size]
    train_policy_labels = train_policy_labels[:batch_count*batch_size]
    train_value_labels = train_value_labels[:batch_count*batch_size]

    batched_states = np.split(train_states, batch_count)
    batched_policy_labels = np.split(train_policy_labels, batch_count)
    batched_value_labels = np.split(train_value_labels, batch_count)
    
    return batched_states, batched_policy_labels, batched_value_labels
