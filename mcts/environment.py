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
    return self.game_over()

  def game_over(self):
    win = np.ones(5)
    win_diag = np.identity(5)

    for i in range(len(self.board)):
      for j in range(len(self.board[i]) - len(win) + 1):

        similarity = np.sum(win * self.board[i][j : j + len(win)])
        similarity_t = np.sum(win * self.board.T[i][j : j + len(win)])

        if similarity ==  5 or similarity_t == 5:
          return 1
        elif similarity ==  -5 or similarity_t == -5:
          return -1

    for i in range(len(self.board) - len(win) + 1):
      for j in range(len(self.board[i]) - len(win) + 1):

        similarity = np.sum(win_diag * self.board[i : i+len(win_diag), j : j +len(win_diag)])
        similarity_t = np.sum(np.rot90(win_diag) * self.board[i : i+len(win_diag), j : j +len(win_diag)])

        if similarity ==  5 or similarity_t == 5:
          return 1
        elif similarity ==  -5 or similarity_t == -5:
          return -1
    
    if np.any(self.board == 0) is False: # draw, VERY IMPROBABLE
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
