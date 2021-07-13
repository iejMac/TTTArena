# Tic-Tac-Toe 5 ❌🅾️
This game is variation of Tic-Tac-Toe that just adds some complexity so that searching over all games is intractable and some form of intuition definately helps. There is also strategy, opening theory, and many intersting tactics I hope a self-play algorithm will be able to discover and maybe even teach me. It's simple enough to understand right away but not too simple where it gets boring in 2 game.

## Rules 📄:
1. Player with the X token always starts the game.
2. Players alternate taking moves until a terminal state is encountered.
3. A terminal state is a state in which:
  - a player has 5 consecutive tokens vertically, horizontally, or diagonally in which case the player whose token it is wins,
  - there are no more empty positions and no player has won the game in which case it is a tie
4. The board can be any shape given by mxn where m, n > 5 (usually m == n > 10, the better you are the more space you need)

## Setup:
1. Clone the repository into your home directory:
```
cd ~
git clone https://github.com/iejMac/AlphaTTT.git
```
2. Install python modules:
```
pip install -r AlphaTTT/requirements.txt
```
3. Play/Observe some games!

## Usage:
1. Launch game:
```
usage: python game.py [-h] [--board_len BOARD_LEN]

Tool for observing Tic-Tac-Toe 5 games

optional arguments:
  -h, --help            show this help message and exit
  --board_len BOARD_LEN
```
2. Enter player types (name of given solution folder) and corresponding parameters in the command line:
```
-= Player X =-
Player type: human
Enter player name: Maciej
-= Player O =-
Player type: alphazero
Model name: trained_model_3
Optimizer state name: trained_opt_state_3
Would you like to adjust MCTS args? yes
num_simulations: 2000 
alpha: 0.01
c_puct: 4
dirichlet_alpha: 0.3
tau: 0.01
```
3. Observe the game

## Creating a solution:
1. Make a new directory with your solutions name
2. Create your_solution/agent.py with a class that inherits Agent from ./agent.py and implements all of the methods described in it.
3. Test it out using game.py

## Data:
In "./data" there is a collection of games played by humans in the format of alterating move coordinates.

## Contributions 👥: 
Any form of contribution in the form of pull requests are encouraged. Here are some ideas:
- Your own approach/solution as a sub-repository. I'd love to be able to put different algorithms against themselves to see which one is best,
- Optimizations and refactors of existing approaches,
- Visualization tools (something like behavioral_cloning/GUI.py but better), 
- Game data in the same format as in "./data"
