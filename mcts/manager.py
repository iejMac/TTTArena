import argparse
from multiprocessing import Process

from model import ZeroTTT
from trainer import Trainer

model_args = {
  "board_len": 10,
  "lr": 1e-4,
  "weight_decay": 3e-4
}

mcts_args = {
  "num_simulations": 10,
  "alpha": 0.35,
  "c_puct": 4,
  "dirichlet_alpha": 0.3
}

db_args = {
  "max_len": 600,
  "augmentations": ["flip", "rotate"]
}

args = {
  "mcts_args": mcts_args,
  "db_args": db_args,
  "board_len": 10
}

def manage_trainer(trainer, buffer_path):
  while True:
    try:
      trainer.generate_buffer(buffer_path)
    except KeyboardInterrupt:
      break
    except Exception as e:
      print(e)

class Manager:
  def __init__(self, model, args, buffer_path, n_proc):
    self.trainers = [Trainer(model, args) for _ in range(n_proc)]
    self.processes = [Process(target=manage_trainer, args=[tr, buffer_path]) for tr in self.trainers]

  def start(self):
    print("Starting...")
    for proc in self.processes:
      proc.start()

    for proc in self.processes:
      proc.join()
 
def main():
  '''
    Buffer path and n_proc passed as args
  '''

  model = ZeroTTT(brain_path="trained_model_4", opt_path="trained_opt_state_4", args=model_args)
  manager = Manager(model, args, "replay_buffer", 10)
  # manager.start()

if __name__ == "__main__":
  main()

