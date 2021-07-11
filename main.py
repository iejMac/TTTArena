from arena import Arena
from alphazero.agent import ZeroAgent
from human.human import Human

model_args = {
  "board_len": 10,
  "lr": 3e-4,
  "weight_decay": 1e-4
}

mcts_args = {
  "num_simulations": 200,
  "alpha": 0.25,
  "c_puct": 4,
  "dirichlet_alpha": 0.3,
  "tau": 1.0
}

args = {
  "model_args": model_args,
	"mcts_args": mcts_args
}


p1 = ZeroAgent("trained_model_3", "trained_opt_state_3", args)
p2 = Human("Maciej")

a = Arena(p1, p2, board_len=10)

a.play(True)

