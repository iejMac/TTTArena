from environment import Environment

class Arena:
  def __init__(self, x_player, o_player, board_len=10)
    self.env = Environment(board_len)
    self.xp = x_player
    self.op = o_player

  def swap_players(self):
    self.xp, self.op = self.op, self.xp

  def play(self, render=True):

    game_state = 10
    current_player, other_player = self.xp, self.op # x always starts

    while game_state == 10:
      move = current_player.make_move(self.env.board) # player makes move
      other_player.update_move(move) # other player gets informed what move was made

      game_state = env.step(move)
      current_player, other_player = other_player, current_player

      if render:
        self.env.render()

    self.env.reset()
    

