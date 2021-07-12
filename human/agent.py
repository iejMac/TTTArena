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
    while True:
      move_str = input("Input a move y, x: ")
      try:
        move = tuple(int(x) for x in move_str.split(","))
      except ValueError:
        print("Incorrect format, try again.")
        continue

      if move[0] < 0 or move[0] >= len(state) or move[1] < 0 or move[1] >= len(state):
        print("Move out of bounds, try again.")
        continue
      if state[move[0]][move[1]] != 0:
        print("Space already taken, try again.") 
        continue
      break
    return move

  def update_state(self, move):
    pass

  @staticmethod
  def get_params():
    name = input("Enter player name: ")
    controls = "keyboard" if sys.platform == "darwin" else "mouse"
    return (name, controls)

'''
      if self.controls == "mouse":
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.pos_x, self.pos_y = mouse_x//self.cell_size, mouse_y//self.cell_size
      elif self.controls == "keyboard":
        pass

      pygame.draw.rect(self.screen, (105, 105, 105), (1 + 30*mouse_x, 1 + 30*mouse_y, 29, 29))

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and mouse_x < self.board_len and mouse_y < self.board_len:
          data.append((mouse_x, mouse_y))



'''
