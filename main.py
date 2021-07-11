from arena import Arena
from alphazero.agent import ZeroAgent
from human.human import Human

model_args = {
  "board_len": 10,
  "lr": 3e-4,
  "weight_decay": 1e-4
}

mcts_args = {
  "num_simulations": 2000,
  "alpha": 0.01,
  "c_puct": 4,
  "dirichlet_alpha": 0.3,
  "tau": 0.01
}

args = {
  "model_args": model_args,
	"mcts_args": mcts_args
}


p1 = ZeroAgent("trained_model_2", "trained_opt_state_2", args)
p2 = Human("Maciej")

a = Arena(p1, p2, board_len=10)

a.play(True)

