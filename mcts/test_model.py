from model import ZeroTTT
from environment import split_state
from environment import Environment

model = ZeroTTT("best_model", "best_opt_state")
model.brain.eval()

env = Environment(board_len=10)

# Position 1
env.step((5, 5))
env.step((0, 1))
env.step((5, 4))
env.step((9, 9))
env.step((5, 6))
env.step((0, 5))
env.step((5, 3))
env.step((5, 0))

env.render()

p, v = model.predict(split_state(env.board))

print(p)

