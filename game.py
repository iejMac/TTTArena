import pygame

def drawgrid(w, rows, surface):
  sizeBtwn = w//rows
  x = 0
  y = 0
  for l in range(rows + 1):
    pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, w))
    pygame.draw.line(surface, (0, 0, 0), (0, y), (w, y))
    x += sizeBtwn
    y += sizeBtwn

pygame.init()

screen = pygame.display.set_mode([800, 500])

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      tunning = False

  screen.fill((0, 0, 0))

  pygame.draw.rect(screen, (255, 255, 255), (5, 5, 20, 10))

  pygame.display.flip()

pygame.quit()
