import numpy as np

class Environment():
  def __init__(self, board_len=30):

    self.board_len = board_len
    self.board = np.zeros((board_len, board_len))

    self.x_token = 1
    self.o_token = -1

    self.turn = self.x_token # x starts

    self.move_hist = []

  def reset(self):
    self.board = np.zeros((self.board_len, self.board_len))
    self.move_hist = []
    self.turn = self.x_token
    return

  def step(self, action, override_turn=None):

    if override_turn is not None:
      self.board[action[0]][action[1]] = override_turn
      self.move_hist.append((action, override_turn))
      return

    self.board[action[0]][action[1]] = self.turn
    self.move_hist.append((action, self.turn))
    self.turn *= -1 # turn swaps
    return self.game_over(self.board)

  @staticmethod
  def game_over(board):
    win = np.ones(5)
    win_diag = np.identity(5)

    for i in range(len(board)):
      for j in range(len(board[i]) - len(win) + 1):

        similarity = np.sum(win * board[i][j : j + len(win)])
        similarity_t = np.sum(win * board.T[i][j : j + len(win)])

        if similarity ==  5 or similarity_t == 5:
          return 1
        elif similarity ==  -5 or similarity_t == -5:
          return -1

    for i in range(len(board) - len(win) + 1):
      for j in range(len(board[i]) - len(win) + 1):

        similarity = np.sum(win_diag * board[i : i+len(win_diag), j : j +len(win_diag)])
        similarity_t = np.sum(np.rot90(win_diag) * board[i : i+len(win_diag), j : j +len(win_diag)])

        if similarity ==  5 or similarity_t == 5:
          return 1
        elif similarity ==  -5 or similarity_t == -5:
          return -1
    
    if np.any(board == 0) is False: # draw
      return 0

    return 10

  def render(self):
    show_board = np.full((self.board_len, self.board_len), ' ')
    for move in self.move_hist:
      action, turn = move
      if turn == 1:
        show_board[action[0]][action[1]] = 'X'
      else:
        show_board[action[0]][action[1]] = 'O'
    print("="*120)    
    print(show_board)
    print("="*120)    
    return
