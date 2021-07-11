import os
import pygame
import inspect
import importlib

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

def drawgrid(w, rows, surface):
  sizeBtwn = w//rows
  x = 0
  y = 0
  for l in range(rows + 1):
    pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, w))
    pygame.draw.line(surface, (0, 0, 0), (0, y), (w, y))
    x += sizeBtwn
    y += sizeBtwn

def play():
  pygame.init()

  screen = pygame.display.set_mode([800, 500])

  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 255, 255), (5, 5, 20, 10))

    pygame.display.flip()

  pygame.quit()

if __name__ == "__main__":
  p1 = get_agent()
  p2 = get_agent()

  a = Arena(p1, p2, board_len=10)
