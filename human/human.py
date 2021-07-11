import sys

sys.path.append('..')
from player import Player

class Human(Player):
  def __init__(self):
    pass
  def init_board(self, board)
    pass
  def make_move(board):
    while True:
      move_str = input("Input a move y, x: ")
      try:
        move = tuple(int(x) for x in move_str.split(","))
      except:
        print("Incorrect format, try again.")

      if move[0] < 0 or move[0] > len(board) or move[1] < 0 or move[1] > len(board):
        print("Move out of bounds, try again.")
        continue
      if board[move[0]][move[1]] != 0:
        print("Space already taken, try again.") 
        continue
    
    break

  def update_move(move):
    pass
