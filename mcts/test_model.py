import numpy as np

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

p, v = model.predict(split_state(env.board))
p = p.detach().cpu().numpy()

env.render()
print(np.around(p, 3))
print(v)

env.reset()

# Position 2
env.step((0, 0))
env.step((5, 5))
env.step((5, 0))
env.step((5, 4))
env.step((0, 9))
env.step((5, 3))
env.step((7, 1))
env.step((5, 6))

env.step((9, 0))

p, v = model.predict(split_state(env.board))
p = p.detach().cpu().numpy()

env.render()
print(np.around(p, 3))
print(v)

env.reset()

# Position 3
env.step((0, 0))
env.step((6, 5))
env.step((5, 0))
env.step((8, 4))
env.step((8, 9))
env.step((5, 9))
env.step((7, 1))
env.step((5, 6))

p, v = model.predict(split_state(env.board))
p = p.detach().cpu().numpy()

env.render()
print(np.around(p, 3))
print(v)

env.reset()

# Position 4
env.step((0, 0))
env.step((6, 5))
env.step((0, 1))
env.step((8, 4))
env.step((0, 2))
env.step((5, 9))
env.step((0, 3))
env.step((5, 6))

p, v = model.predict(split_state(env.board))
p = p.detach().cpu().numpy()

env.render()
print(np.around(p, 3))
print(v)

