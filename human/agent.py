import os
import sys
import pygame

sys.path.append(os.path.join(os.environ["HOME"], "AlphaTTT"))

from agent import Agent

class Human(Agent):
  def __init__(self, name, controls):
    super().__init__(name)
    self.controls = controls

  def reset(self, state):
    pass

  def make_action(self, state):
    cell_size = 30 # get this from main game.py file somehow (global var)
    chosen = False
    while not chosen:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return None
        if event.type == pygame.MOUSEBUTTONDOWN:
          mouse_x, mouse_y = pygame.mouse.get_pos()
          pos_x, pos_y = mouse_x//cell_size, mouse_y//cell_size
          if state[pos_y][pos_x] != 0:
            continue
          chosen = True
    return (pos_y, pos_x)

  def update_state(self, move):
    pass

  @staticmethod
  def get_params():
    name = input("Enter player name: ")
    controls = "keyboard" if sys.platform == "darwin" else "mouse"
    return (name, controls)
