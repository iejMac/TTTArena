from environment import Environment

class Arena:
  def __init__(self, x_agent, o_agent, board_len=10):
    self.env = Environment(board_len)
    self.xa = x_agent
    self.oa = o_agent

    self.xa.reset(self.env.board)
    self.oa.reset(self.env.board)

    self.game_state = 10
    self.ready_for_move = True

  def reset(self):
    self.game_state = 10

    self.env.reset()
    self.xa.reset(self.env.board)
    self.oa.reset(self.env.board)

  def is_active(self):
    return self.game_state == 10

  def swap_agents(self):
    self.xa, self.oa = self.oa, self.xa

  def move(self):
    self.ready_for_move = False
    current_player, other_player = (self.xa, self.oa) if self.env.turn == 1 else (self.oa, self.xa)
    move = current_player.make_action(self.env.board)
    if move is None:
      return False
    other_player.update_state(move)
    self.game_state = self.env.step(move)
    self.ready_for_move = True

    if self.game_state != 10:
      print(f"{current_player.name} won in {len(self.env.move_hist)} moves") 

    return True
