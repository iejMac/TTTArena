from arena import Arena
from human.human import Human

a = Arena(Human(), Human(), board_len=10)

a.play(True)

