from environment import Environment

class Arena:
  def __init__(self, x_agent, o_agent, board_len=10):
    self.env = Environment(board_len)
    self.xa = x_agent
    self.oa = o_agent

  def swap_agents(self):
    self.xa, self.oa = self.oa, self.xa

  def play(self, render=True):

    self.xa.init_state(self.env.board)
    self.oa.init_state(self.env.board)

    game_state = 10
    current_player, other_player = self.xa, self.oa # x always starts

    while game_state == 10:
      move = current_player.make_action(self.env.board) # player makes move
      other_player.update_state(move) # other player gets informed what move was made

      game_state = self.env.step(move)
      current_player, other_player = other_player, current_player

      if render:
        self.env.render()

    print(f"{other_player.name} won")
    self.env.reset()
