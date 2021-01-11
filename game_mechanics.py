# Description:
# Functions that deal with gameplay (rules, environment creation, space selection etc.)

# Imports:
import numpy as np
import random
import environment as env
import model_new as model


# Constants:
board_len = 30

# Functions:


def create_board():
    board = np.zeros((board_len, board_len))
    return board


def select_space(board, player, position):
    # Makes the move and returns a boolean which states if the move was successfully made
    if player == 'X':
        symbol = 2
    else:
        symbol = 1

    if board[position[0]][position[1]] == 0:
        board[position[0]][position[1]] = symbol
        return True
    else:
        return False


def game_over(board):
    # This functions returns 2 if X wins, 1 if O wins, 0 if it's a draw,
    # and -1 if the game is still ongoing.

    symbols = [1, 2]

    # Checking rows for wins
    for i, row in enumerate(board):
        len_construct = 0
        continuous = False
        sym = 0
        for j, space in enumerate(row):
            if space in symbols:
                if not continuous:
                    len_construct += 1
                    continuous = True
                    sym = space
                else:
                    if space == sym:
                        len_construct += 1
                    else:
                        len_construct = 1
                        sym = space
            else:
                len_construct = 0
                sym = 0
                continuous = False

            if len_construct == 5:
                return sym

    # Do the same algorithm on a rotated board making it check columns
    board = board.T

    for i, row in enumerate(board):
        len_construct = 0
        continuous = False
        sym = 0
        for j, space in enumerate(row):
            if space in symbols:
                if not continuous:
                    len_construct += 1
                    continuous = True
                    sym = space
                else:
                    if space == sym:
                        len_construct += 1
                    else:
                        len_construct = 1
                        sym = space
            else:
                len_construct = 0
                sym = 0
                continuous = False

            if len_construct == 5:
                return sym

    # Diagonals setup:
    b_side = len(board)

    board_diag = np.zeros((2 * b_side - 1) * b_side)
    board_diag = np.array([i for i in board_diag])
    board_diag = board_diag.reshape((2 * b_side - 1, b_side))
    diagonal_lengths_1 = []
    diagonal_lengths_2 = []

    for i in range(int(b_side - 1)):
        diagonal_lengths_1.append(i + 1)
    for i in range(int(b_side)):
        diagonal_lengths_2.append(int(b_side - i))

    for i, current_len in enumerate(diagonal_lengths_1):
        for j in range(current_len):
            board_diag[i][j] = board[j][b_side - 1 - i + j]

    for i, current_len in enumerate(diagonal_lengths_2):
        for j in range(current_len):
            board_diag[len(diagonal_lengths_1) + i][j] = board[i + j][j]

    # Repeat algorithm on array of diagonals:

    for i, row in enumerate(board_diag):
        len_construct = 0
        continuous = False
        sym = 0
        for j, space in enumerate(row):
            if space in symbols:
                if not continuous:
                    len_construct += 1
                    continuous = True
                    sym = space
                else:
                    if space == sym:
                        len_construct += 1
                    else:
                        len_construct = 1
                        sym = space
            else:
                len_construct = 0
                sym = 0
                continuous = False
            if len_construct == 5:
                return sym

    # Rotate diagonal board and repeat

    board = np.rot90(board)
    board_diag = np.zeros((2 * b_side - 1) * b_side)
    board_diag = np.array([i for i in board_diag])
    board_diag = board_diag.reshape((2 * b_side - 1, b_side))
    diagonal_lengths_1 = []
    diagonal_lengths_2 = []

    for i in range(int(b_side - 1)):
        diagonal_lengths_1.append(i + 1)
    for i in range(int(b_side)):
        diagonal_lengths_2.append(int(b_side - i))

    for i, current_len in enumerate(diagonal_lengths_1):
        for j in range(current_len):
            board_diag[i][j] = board[j][b_side - 1 - i + j]

    for i, current_len in enumerate(diagonal_lengths_2):
        for j in range(current_len):
            board_diag[len(diagonal_lengths_1) + i][j] = board[i + j][j]

    for i, row in enumerate(board_diag):
        len_construct = 0
        continuous = False
        sym = 0
        for j, space in enumerate(row):
            if space in symbols:
                if not continuous:
                    len_construct += 1
                    continuous = True
                    sym = space
                else:
                    if space == sym:
                        len_construct += 1
                    else:
                        len_construct = 1
                        sym = space
            else:
                len_construct = 0
                sym = 0
                continuous = False
            if len_construct == 5:
                return sym

    if np.any(0 in board):
        return -1

    return 0


def play_n_games(playerX=None, playerO=None, n=1, render=False):

    environment = env.TTTenvironment()

    X_agent = model.AlphaTTT('X', policy_name=playerX)
    O_agent = model.AlphaTTT('O', policy_name=playerO)

    # Metrics to note:
    x_win_count = 0
    o_win_count = 0

    # Hyperparameters:
    # initial_epsilon = 0.05
    initial_epsilon = 0

    for game in range(n):
        print(f'Game #{game + 1}')
        turn = 'X'
        first_move = True
        winner = None
        game_reward = 0
        move_count = 0
        obs, done = environment.reset(), False
        # X_agent.clear_memory()
        # O_agent.clear_memory()
        while not done:

            if turn == 'X':
                # X_agent.memorize_state(obs)
                action, was_random = X_agent.policy_predict_action(obs, initial_epsilon, first_move=first_move)
                if first_move is True:
                    first_move = False
            else:
                # O_agent.memorize_state(obs)
                action, was_random = O_agent.policy_predict_action(obs, initial_epsilon)

            prev_obs = obs.copy()

            obs, reward, done, info = environment.step(turn, action, was_random)
            move_count += 1

            game_reward += reward

            winner = info

            # Swap turns:
            if turn == 'X':
                turn = 'O'
            else:
                turn = 'X'

        print(f'Move count: {move_count}, Game reward: {game_reward}, Winner: {winner}')
        if winner == 'X':
            x_win_count += 1
        elif winner == 'O':
            o_win_count += 1

        if render is True:
            environment.render(show_randomness=True)

    x_win_ratio = x_win_count / (x_win_count + o_win_count)
    o_win_ratio = o_win_count / (x_win_count + o_win_count)

    print(f'Final statistics: X won {x_win_ratio} of games, O won {o_win_ratio} of games.')


def inscribe_board(board, rim_width):
    side = len(board) + 2*rim_width
    new_board = np.zeros((side*side))
    new_board = np.array([i for i in new_board]).reshape((side, side))

    for i, row in enumerate(board):
        for j, space in enumerate(row):
            new_board[rim_width + i][rim_width+j] = space

    return new_board


def available_moves(board):
    move_list = []
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == 0:
                move_list.append((i, j))
    return move_list


def generate_availability_mask(board, flatten=False):
    mask = create_board()
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell == 0:
                select_space(mask, 'O', (i, j))

    if flatten is True:
        mask = mask.flatten()

    return mask


def evaluate_position(board):
    # Good constructs:
    universal_value = 32

    o_win = [1, 1, 1, 1, 1]
    o_uncapped_4 = [1, 1, 1, 1]
    o_uncapped_3 = [1, 1, 1]
    o_uncapped_2 = [1, 1]
    o_capped_l_4 = [2, 1, 1, 1, 1]
    o_capped_r_4 = [1, 1, 1, 1, 2]
    o_capped_l_3 = [2, 1, 1, 1]
    o_capped_r_3 = [1, 1, 1, 2]
    o_capped_l_2 = [2, 1, 1]
    o_capped_r_2 = [1, 1, 2]

    o_constructs = [o_win, o_uncapped_4, o_uncapped_3, o_uncapped_2, o_capped_l_4, o_capped_r_4, o_capped_l_3,
                    o_capped_r_3, o_capped_l_2, o_capped_r_2]

    o_value_dict = {
        tuple(o_win): -20 * universal_value,
        tuple(o_uncapped_4): -universal_value,
        tuple(o_uncapped_3): -universal_value / 4,
        tuple(o_uncapped_2): -universal_value / 16,
        tuple(o_capped_l_4): -universal_value / 2,
        tuple(o_capped_r_4): -universal_value / 2,
        tuple(o_capped_l_3): -universal_value / 8,
        tuple(o_capped_r_3): -universal_value / 8,
        tuple(o_capped_l_2): -universal_value / 32,
        tuple(o_capped_r_2): -universal_value / 32
    }

    x_win = [2, 2, 2, 2, 2]
    x_uncapped_4 = [2, 2, 2, 2]
    x_uncapped_3 = [2, 2, 2]
    x_uncapped_2 = [2, 2]
    x_capped_l_4 = [1, 2, 2, 2, 2]
    x_capped_r_4 = [2, 2, 2, 2, 1]
    x_capped_l_3 = [1, 2, 2, 2]
    x_capped_r_3 = [2, 2, 2, 1]
    x_capped_l_2 = [1, 2, 2]
    x_capped_r_2 = [2, 2, 1]

    x_constructs = [x_win, x_uncapped_4, x_uncapped_3, x_uncapped_2, x_capped_l_4, x_capped_r_4, x_capped_l_3,
                    x_capped_r_3, x_capped_l_2, x_capped_r_2]

    x_value_dict = {
        tuple(x_win): 20 * universal_value,
        tuple(x_uncapped_4): universal_value,
        tuple(x_uncapped_3): universal_value / 4,
        tuple(x_uncapped_2): universal_value / 16,
        tuple(x_capped_l_4): universal_value / 2,
        tuple(x_capped_r_4): universal_value / 2,
        tuple(x_capped_l_3): universal_value / 8,
        tuple(x_capped_r_3): universal_value / 8,
        tuple(x_capped_l_2): universal_value / 32,
        tuple(x_capped_r_2): universal_value / 32
    }

    evaluation = 0

    symbols = [2, 1]

    # Check rows for constructs:
    for i, row in enumerate(board):
        len_construct = 0
        copying_construct = False
        current_symbol = 3
        analyze = False
        construct_for_analysis = []

        for j, space in enumerate(row):
            if space in symbols:

                len_construct += 1
                if copying_construct is True and space != current_symbol:
                    if len_construct > 2:
                        construct_for_analysis = list(row[j - len_construct + 1:j + 1])
                        len_construct = 2
                        analyze = True
                    else:
                        len_construct = 2
                        analyze = False

                if copying_construct is False:
                    copying_construct = True
                current_symbol = space

            else:
                if copying_construct is True:
                    copying_construct = False
                    if 1 < len_construct < 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        analyze = True
                    elif len_construct >= 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        if construct_for_analysis[0] != construct_for_analysis[1]:
                            construct_for_analysis = construct_for_analysis[1:6]
                        # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                        #     construct_for_analysis = construct_for_analysis[0:5]
                        else:
                            construct_for_analysis = construct_for_analysis[0:5]

                        analyze = True

                    len_construct = 0
                    current_symbol = 3

            if len(construct_for_analysis) >= 6:
                if construct_for_analysis[0] != construct_for_analysis[1] and construct_for_analysis[-1] != \
                        construct_for_analysis[-2] and construct_for_analysis[0] == construct_for_analysis[-1]:
                    if len(construct_for_analysis[1:-1]) > 4:
                        construct_for_analysis = construct_for_analysis[1:6]
                    else:
                        construct_for_analysis = construct_for_analysis[-2:]

                    len_construct = 2
                else:
                    if construct_for_analysis[0] != construct_for_analysis[1]:
                        construct_for_analysis = construct_for_analysis[1:6]
                    # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                    #     construct_for_analysis = construct_for_analysis[0:5]
                    else:
                        construct_for_analysis = construct_for_analysis[0:5]

            if analyze is True:
                if construct_for_analysis in o_constructs:
                    evaluation += o_value_dict[tuple(construct_for_analysis)]

                elif construct_for_analysis in x_constructs:
                    evaluation += x_value_dict[tuple(construct_for_analysis)]

                analyze = False

    board = board.T

    # Check columns for constructs:
    for i, row in enumerate(board):
        len_construct = 0
        copying_construct = False
        current_symbol = 3
        analyze = False
        construct_for_analysis = []

        for j, space in enumerate(row):
            if space in symbols:

                len_construct += 1

                if copying_construct is True and space != current_symbol:
                    if len_construct > 2:
                        construct_for_analysis = list(row[j - len_construct + 1:j + 1])
                        len_construct = 2
                        analyze = True
                    else:
                        len_construct = 2
                        analyze = False

                if copying_construct is False:
                    copying_construct = True

                current_symbol = space

            else:
                if copying_construct is True:
                    copying_construct = False
                    if 1 < len_construct < 6:

                        construct_for_analysis = list(row[j - len_construct:j])
                        analyze = True
                    elif len_construct >= 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        if construct_for_analysis[0] != construct_for_analysis[1]:
                            construct_for_analysis = construct_for_analysis[1:6]
                        # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                        #     construct_for_analysis = construct_for_analysis[0:5]
                        else:
                            construct_for_analysis = construct_for_analysis[0:5]
                        analyze = True

                    len_construct = 0
                    current_symbol = 3

            if len(construct_for_analysis) >= 6:
                if construct_for_analysis[0] != construct_for_analysis[1] and construct_for_analysis[-1] != \
                        construct_for_analysis[-2] and construct_for_analysis[0] == construct_for_analysis[-1]:
                    if len(construct_for_analysis[1:-1]) > 4:
                        construct_for_analysis = construct_for_analysis[1:6]
                    else:
                        construct_for_analysis = construct_for_analysis[-2:]
                    len_construct = 2
                else:
                    if construct_for_analysis[0] != construct_for_analysis[1]:
                        construct_for_analysis = construct_for_analysis[1:6]
                    # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                    #     construct_for_analysis = construct_for_analysis[0:5]
                    else:
                        construct_for_analysis = construct_for_analysis[0:5]

            if analyze is True:
                if construct_for_analysis in o_constructs:
                    evaluation += o_value_dict[tuple(construct_for_analysis)]

                elif construct_for_analysis in x_constructs:
                    evaluation += x_value_dict[tuple(construct_for_analysis)]

                analyze = False

    board = board.T

    # Check diagonals for constructs:
    # First diagonal (top-left to bottom-right):

    b_side = len(board)

    board_diag = np.zeros((2 * b_side - 1) * b_side)
    # board_diag = np.array([str(i) for i in board_diag])
    board_diag = board_diag.reshape((2 * b_side - 1, b_side))
    diagonal_lengths_1 = []
    diagonal_lengths_2 = []

    for i in range(int(b_side - 1)):
        diagonal_lengths_1.append(i + 1)
    for i in range(int(b_side)):
        diagonal_lengths_2.append(int(b_side - i))

    for i, current_len in enumerate(diagonal_lengths_1):
        for j in range(current_len):
            board_diag[i][j] = board[j][b_side - 1 - i + j]

    for i, current_len in enumerate(diagonal_lengths_2):
        for j in range(current_len):
            board_diag[len(diagonal_lengths_1) + i][j] = board[i + j][j]

    for i, row in enumerate(board_diag):
        len_construct = 0
        copying_construct = False
        current_symbol = 3
        analyze = False
        construct_for_analysis = []

        for j, space in enumerate(row):
            if space in symbols:

                len_construct += 1

                if copying_construct is True and space != current_symbol:
                    if len_construct > 2:
                        construct_for_analysis = list(row[j - len_construct + 1:j + 1])
                        len_construct = 2
                        analyze = True
                    else:
                        len_construct = 2
                        analyze = False

                if copying_construct is False:
                    copying_construct = True

                current_symbol = space

            else:
                if copying_construct is True:
                    copying_construct = False
                    if 1 < len_construct < 6:

                        construct_for_analysis = list(row[j - len_construct:j])
                        analyze = True
                    elif len_construct >= 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        if construct_for_analysis[0] != construct_for_analysis[1]:
                            construct_for_analysis = construct_for_analysis[1:6]
                        # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                        #     construct_for_analysis = construct_for_analysis[0:5]
                        else:
                            construct_for_analysis = construct_for_analysis[0:5]
                        analyze = True

                    len_construct = 0
                    current_symbol = 3

            if len(construct_for_analysis) >= 6:
                if construct_for_analysis[0] != construct_for_analysis[1] and construct_for_analysis[-1] != \
                        construct_for_analysis[-2] and construct_for_analysis[0] == construct_for_analysis[-1]:
                    if len(construct_for_analysis[1:-1]) > 4:
                        construct_for_analysis = construct_for_analysis[1:6]
                    else:
                        construct_for_analysis = construct_for_analysis[-2:]
                    len_construct = 2
                else:
                    if construct_for_analysis[0] != construct_for_analysis[1]:
                        construct_for_analysis = construct_for_analysis[1:6]
                    # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                    #     construct_for_analysis = construct_for_analysis[0:5]
                    else:
                        construct_for_analysis = construct_for_analysis[0:5]

            if analyze is True:
                if construct_for_analysis in o_constructs:
                    evaluation += o_value_dict[tuple(construct_for_analysis)]
                elif construct_for_analysis in x_constructs:
                    evaluation += x_value_dict[tuple(construct_for_analysis)]
                analyze = False

    # Second diagonal (bottom-left to top_right):

    board = np.rot90(board)

    board_diag = np.zeros((2 * b_side - 1) * b_side)
    # board_diag = np.array([str(i) for i in board_diag])
    board_diag = board_diag.reshape((2 * b_side - 1, b_side))

    diagonal_lengths_1 = []
    diagonal_lengths_2 = []

    for i in range(int(b_side - 1)):
        diagonal_lengths_1.append(i + 1)
    for i in range(int(b_side)):
        diagonal_lengths_2.append(int(b_side - i))

    for i, current_len in enumerate(diagonal_lengths_1):
        for j in range(current_len):
            board_diag[i][j] = board[j][b_side - 1 - i + j]

    for i, current_len in enumerate(diagonal_lengths_2):
        for j in range(current_len):
            board_diag[len(diagonal_lengths_1) + i][j] = board[i + j][j]

    for i, row in enumerate(board_diag):
        len_construct = 0
        copying_construct = False
        current_symbol = 3
        analyze = False
        construct_for_analysis = []

        for j, space in enumerate(row):
            if space in symbols:

                len_construct += 1

                if copying_construct is True and space != current_symbol:
                    if len_construct > 2:
                        construct_for_analysis = list(row[j - len_construct + 1:j + 1])
                        len_construct = 2
                        analyze = True
                    else:
                        len_construct = 2
                        analyze = False

                if copying_construct is False:
                    copying_construct = True

                current_symbol = space

            else:
                if copying_construct is True:
                    copying_construct = False
                    if 1 < len_construct < 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        analyze = True
                    elif len_construct >= 6:
                        construct_for_analysis = list(row[j - len_construct:j])
                        if construct_for_analysis[0] != construct_for_analysis[1]:
                            construct_for_analysis = construct_for_analysis[1:6]
                        # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                        #     construct_for_analysis = construct_for_analysis[0:5]
                        else:
                            construct_for_analysis = construct_for_analysis[0:5]
                        analyze = True

                    len_construct = 0
                    current_symbol = 3

            if len(construct_for_analysis) >= 6:
                if construct_for_analysis[0] != construct_for_analysis[1] and construct_for_analysis[-1] != \
                        construct_for_analysis[-2] and construct_for_analysis[0] == construct_for_analysis[-1]:
                    if len(construct_for_analysis[1:-1]) > 4:
                        construct_for_analysis = construct_for_analysis[1:6]
                    else:
                        construct_for_analysis = construct_for_analysis[-2:]
                    len_construct = 2
                else:
                    if construct_for_analysis[0] != construct_for_analysis[1]:
                        construct_for_analysis = construct_for_analysis[1:6]
                    # elif construct_for_analysis[-1] != construct_for_analysis[-2]:
                    #     construct_for_analysis = construct_for_analysis[0:5]
                    else:
                        construct_for_analysis = construct_for_analysis[0:5]

            if analyze is True:
                if construct_for_analysis in o_constructs:
                    evaluation += o_value_dict[tuple(construct_for_analysis)]
                elif construct_for_analysis in x_constructs:
                    evaluation += x_value_dict[tuple(construct_for_analysis)]
                analyze = False

    return evaluation


def swap_symbols(board):
    for i, row in enumerate(board):
        for j, space in enumerate(row):
            if space == 2:
                board[i][j] = 1
            elif space == 1:
                board[i][j] = 2
