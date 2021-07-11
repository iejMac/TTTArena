from model import ZeroTTT
from trainer import Trainer

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

db_args = {
  "max_len": 2000,
  "augmentations": ["flip", "rotate"]
}

model = ZeroTTT(brain_path=None, opt_path=None, args=model_args)

args = {
  "mcts_args": mcts_args,
  "db_args": db_args,
  "board_len": 10
}

trainer = Trainer(model, args)
trainer.generate_game(True)
