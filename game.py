import pygame

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
