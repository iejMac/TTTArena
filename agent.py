class Agent:
  def __init__(self, name):
    self.name = name
  def reset(self, state):
    # Completely resets the state of the Agent for a new game
    return
  def make_action(self, state):
    # Returns a valid move in (row, column) format where 0 <= row, column < board_len
    move = (0, 0)
    return move
  def update_state(self, move):
    # Update the internal state of an agent according to the move made by the opponent (if necessary)
    return
  @staticmethod
  def get_params():
    # Get agent parameters from command line input and return in tuple form
    return ()
