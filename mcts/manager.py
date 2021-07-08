import argparse
import numpy as np
from multiprocessing import Process

from model import ZeroTTT
from trainer import Trainer

model_args = {
  "board_len": 10,
  "lr": 1e-4,
  "weight_decay": 3e-4
}

mcts_args = {
  "num_simulations": 200,
  "alpha": 0.25,
  "c_puct": 4,
  "dirichlet_alpha": 0.3
}

db_args = {
  "max_len": 10000,
  "augmentations": ["flip", "rotate"]
}

args = {
  "mcts_args": mcts_args,
  "db_args": db_args,
  "board_len": 10
}

def manage_trainer(model_name, opt_state_name, model_args, trainer_args, buffer_path, seed):
  np.random.seed(seed)
  model = ZeroTTT(model_name, opt_state_name, model_args)
  trainer = Trainer(model, args)
  while True:
    try:
      trainer.generate_buffer(buffer_path)
    except KeyboardInterrupt:
      break
    except Exception as e:
      print(e)

class Manager:
  def __init__(self, model_name, opt_state_name, model_args, trainer_args, buffer_path, n_proc):
    if model_name == "None":
        model_name = None
    if opt_state_name == "None":
        opt_state_name = None

    self.processes = [Process(target=manage_trainer, args=(model_name, opt_state_name, model_args, trainer_args, buffer_path, nr)) for nr in range(n_proc)]

  def start(self):
    print("Starting...")
    for proc in self.processes:
      proc.start()

    for proc in self.processes:
      proc.join()

# TODO: add model/mcts args update with optional args in argparse
def get_arg_parser():
  parser = argparse.ArgumentParser(description="Manage multiple Trainers generating a replay buffer")
  parser.add_argument("model_name", type=str, help="Name of model stored in AlphaTTT/mcts/models")
  parser.add_argument("opt_state_name", type=str, help="Name of optimizer state stored in AlphaTTT/mcts/models")
  parser.add_argument("n_trainers", type=int, help="Number of trainers")
  parser.add_argument("buffer_path", type=str, help="Path to replay buffer")

  return parser

def main():
  parser = get_arg_parser()
  manager_args = parser.parse_args()

  manager = Manager(manager_args.model_name, manager_args.opt_state_name, model_args, args, manager_args.buffer_path, manager_args.n_trainers)
  manager.start()

if __name__ == "__main__":
  main()
