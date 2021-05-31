# Tic-Tac-Toe 5 ❌🅾️
This game is variation of Tic-Tac-Toe that just adds some complexity so that searching over all games is intractable and some form of intuition definately helps. There is also strategy, opening theory, and many intersting tactics I hope a self-play algorithm will be able to discover and maybe even teach me. It's simple enough to understand right away but not too simple where it gets boring in 2 game.

## Rules 📄:
1. Player with the X token always starts the game.
2. Players alternate taking moves until a terminal state is encountered.
3. A terminal state is a state in which:
  - there are 5 consecutive tokens vertically, horizontally, or diagonally in which case the player whose token it is wins,
  - there are no more empty positions and no player has won the game in which case it is a tie
4. The board can be any shape given by mxn where m, n > 5 (usually m == n > 10, the better you are the more space you need)

## Data ⌗::
In "./data" there is a collection of games played by humans in the format of alterating move coordinates. Since X always starts this is a nice compact way to represent a complete game.

## Contributions 👥: 
Any form of contribution in the form of pull requests are encouraged. Here are some ideas:
- Your own approach/solution as a sub-repository. I'd love to be able to put different algorithms against themselves to see which one is best,
- Optimizations and refactors of existing approaches,
- Visualization tools (something like behavioral_cloning/GUI.py but better), 
- Game data in the same format as in "./data"
