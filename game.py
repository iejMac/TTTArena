import os
import pygame
import inspect
import argparse
import importlib

from multiprocessing import Process

from arena import Arena

def get_agent():
  # Get agent type
  p_type = input("Player type: ")
  agent_mod = importlib.import_module(p_type + ".agent")
  for name, cls in inspect.getmembers(agent_mod, inspect.isclass):
    base_names = [base.__name__ for base in cls.__bases__]
    if "Agent" in base_names:
      agent_cls = cls
      break

  # Get params
  params = agent_cls.get_params()
  agent = agent_cls(*params)
  return agent

def get_arg_parser():
  parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe 5")
  parser.add_argument("--board_len", type=int, default=10)
  return parser

class Game:
  def __init__(self, board_len):
    self.cell_size = 30 # constant cell size
    self.board_len = board_len

    self.pos_x, self.pos_y = self.board_len//2, self.board_len//2

  def draw_board(self):
    # Grid:
    w = self.board_len * self.cell_size
    x, y = 0, 0
    for _ in range(self.board_len + 1):
      pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, w))
      pygame.draw.line(self.screen, (0, 0, 0), (0, y), (w, y))
      x += self.cell_size
      y += self.cell_size

    # Tokens:
    for i, coords in enumerate(self.arena.env.move_hist):
      self.screen.blit(self.xo_textsurface[i%2], (coords[1]*self.cell_size + self.cell_size//7, coords[0]*self.cell_size + self.cell_size//14))

  def get_players(self):
    print("-= Player X =-")
    self.xp = get_agent()
    print("-= Player O =-")
    self.op = get_agent()
    
  def play_game(self):

    pygame.init()
    clock = pygame.time.Clock()
    myfont = pygame.font.SysFont('courier new', 1.5*self.cell_size)
    self.xo_textsurface = [myfont.render('X', False, (0, 0, 0)), myfont.render('O', False, (0, 0, 0))]

    screen_len = self.board_len * self.cell_size
    self.screen = pygame.display.set_mode([screen_len, screen_len])
    self.arena = Arena(self.xp, self.op, board_len=self.board_len)
    pygame.display.update() # hack for grid to appear before move
    clock.tick(60)

    running = True
    while running:

      self.screen.fill((255, 255, 255))
      self.draw_board()
      pygame.display.update()
      clock.tick(60)

      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      if not self.arena.is_active():
        again = input("Would you like to play again? (y/n): ")
        if again == 'y':
          self.arena.reset()
          continue
        break

      if self.arena.ready_for_move:
        running = self.arena.move()


    pygame.quit()

if __name__ == "__main__":
  parser = get_arg_parser()
  args = parser.parse_args()

  g = Game(args.board_len)

  g.get_players()
  g.play_game()
