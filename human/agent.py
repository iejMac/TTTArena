import os
import sys
import pygame

sys.path.append(os.path.join(os.environ["HOME"], "TTTArena"))

from agent import Agent

class Human(Agent):
  def __init__(self, name):
    super().__init__(name)

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

  @staticmethod
  def get_params():
    name = input("Enter player name: ")
    return (name,)
