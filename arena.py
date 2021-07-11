from environment import Environment

class Arena:
  def __init__(self, x_agent, o_agent, board_len=10):
    self.env = Environment(board_len)
    self.xa = x_agent
    self.oa = o_agent

    self.xa.init_state(self.env.board)
    self.oa.init_state(self.env.board)

    self.game_state = 10

  def swap_agents(self):
    self.xa, self.oa = self.oa, self.xa

  def move(self):
    if self.game_state != 10:
      return False

    current_player, other_player = (self.xa, self.oa) if self.env.turn == 1 else (self.oa, self.xa)
    move = current_player.make_action(self.env.board)
    other_player.update_state(move)
    self.game_state = self.env.step(move)
    return True
