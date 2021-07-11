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

def drawdata(data, surface):
  for i, coords in enumerate(data):
    if i%2:
        screen.blit(textsurfaceX, (coords[0]*size + size/3, coords[1]*size))
    else:
        screen.blit(textsurfaceO, (coords[0]*size + size/3, coords[1]*size))

def play():
    pygame.init()

    screen_height = 600
    screen = pygame.display.set_mode([screen_height, screen_height])


    rows = 10
    size = screen_height/rows

    data = []

    myfont = pygame.font.SysFont('courier new', int(3*size/4))
    textsurfaceX = myfont.render('X', False, (0, 0, 0))
    textsurfaceO = myfont.render('O', False, (0, 0, 0))

    running = True
    while running:
      
      (mouse_x, mouse_y) = pygame.mouse.get_pos()
      mouse_x = int(mouse_x/size)
      mouse_y = int(mouse_y/size)
      
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        if event.type == pygame.MOUSEBUTTONDOWN and mouse_x < rows and mouse_y < rows:
          data.append((mouse_x, mouse_y))

      screen.fill((255, 255, 255))

      drawgrid(screen_height, rows, screen)
      drawdata(data, screen)
      
      if mouse_x < rows and mouse_y < rows:
        pygame.draw.rect(screen, (190, 190, 190), (size*mouse_x, size*mouse_y, size, size))
      
      pygame.display.flip()

if __name__ == "__main__":
  p1 = get_agent()
  p2 = get_agent()

  a = Arena(p1, p2, board_len=10)
