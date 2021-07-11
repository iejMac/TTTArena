import sys

sys.path.append('..')
from agent import Agent

class Human(Agent):
  def __init__(self, name):
    super().__init__(name)
  def init_state(self, state):
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
