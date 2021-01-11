# Description:
# Functions that deal with data generation/manipulation


# Imports:
import game_mechanics as gm
import GUI
import numpy as np
from keras.utils.np_utils import to_categorical
from _collections import deque
import os


# Constants:
board_len = 30


# Functions:

def moves_to_board(moves):
    new_board = gm.create_board()
    # symbol_dict = {1: 'O', 2: 'X'}
    player = 'X'
    for move in moves:
        # gm.select_space(new_board, symbol_dict[move[0]], move[1])
        gm.select_space(new_board, player, move)
        if player == 'X':
            player = 'O'
        else:
            player = 'X'

    return new_board

# Functions for first training strategy:
# -----------------------------------------------


def move_to_board(move):
    new_board = gm.create_board()
    gm.select_space(new_board, 'O', move)
    return new_board


def generate_y_data(moves):
    y = []
    for move in moves:
        # Trying to change a little bit so categorical_xentropy works
        # y.append(move_to_board(move))
        y.append(30*move[0] + move[1])

    y = to_categorical(y, num_classes=900)

    return y


def prepare_data(observations, moves):

    moves = generate_y_data(moves)

    X = []
    # y = []
    y = moves
    for obs, mov in zip(observations, moves):
        X.append(obs.reshape((gm.board_len, gm.board_len, 1)))
        # y.append(mov.reshape((gm.board_len, gm.board_len, 1)))

    X = np.array(X)
    # y = np.array(y)

    return X, y


# ------------------------------------------------
# Functions for second training strategy:
# ------------------------------------------------

def make_batches(ar_len, batch_size):
    # ar_len = len(array)
    batches = []
    n_batches = int(ar_len/batch_size)
    for i in range(n_batches):
        # Samples batches from the end of the array (situational because in this game the last gradient is very important)
        # batches.append(array[ar_len - (i+1)*batch_size:ar_len - i*batch_size])
        batches.append((ar_len - (i+1)*batch_size, ar_len - i*batch_size))
    return batches


# ------------------------------------------------

def split_state(state):
    # Making state have 2 channels, one for X's, the other for O's
    transformed_state = np.zeros((2, gm.board_len, gm.board_len))

    for i, row in enumerate(state):
        for j, space in enumerate(row):
            if space != 0:
                transformed_state[int(space-1)][i][j] = 1

    # Output shape is (2, board_len, board_len)
    return transformed_state


def reshape_board(board):
    n_channels = 2
    return board.reshape((gm.board_len, gm.board_len, n_channels))


def reshape_labels(label):
    return label.flatten()


def gen_data(start_nr, n=20):
    x_wins = 0
    o_wins = 0
    for game_nr in range(n):

        game_history = GUI.play_game('pvp', return_history=True)
        if len(game_history) % 2 == 0:
            o_wins += 1
        else:
            x_wins += 1

        game_string = 'policy_net_data/game_' + str(start_nr + game_nr) + '_data.csv'
        np.savetxt(game_string, game_history, delimiter=',')

        print(f'X has won {x_wins} games and O has won {o_wins} games.')
    return


def pull_game_data(game_nr, move_cluster):
    game_hist = np.loadtxt('policy_net_data/game_' + str(game_nr) + '_data.csv', delimiter=',', dtype=int)
    if move_cluster is True:
        translated_game_hists = translate_cluster(game_hist)
    else:
        translated_game_hists = [game_hist]
    return translated_game_hists


def augment_game(game_hists):

    # We need this function to return the augmented data set (state, action)
    # so we can train the policy

    augmented_data = [None for _ in range(16)]
    augmented_labels = [None for _ in range(16)]

    for game_hist in game_hists:

        states = []
        labels = []

        board = gm.create_board()
        empty_board = gm.create_board()

        for i, move in enumerate(game_hist):

            action = (int(move[0]), int(move[1]))

            if i % 2 == 0:
                player = 'X'
            else:
                player = 'O'

            board_copy = board.copy()
            # Marking the move that was made
            board_copy[action[0]][action[1]] = 5

            # Augmentation part
            for k in range(2):
                for m in range(2):
                    for j in range(4):
                        # 4 rotations (because rotationally symmetric)

                        move_made = np.unravel_index(np.argmax(board_copy), (gm.board_len, gm.board_len))

                        empty_board_copy = empty_board.copy()
                        gm.select_space(empty_board_copy, 'O', move_made)
                        labels.append(reshape_labels(empty_board_copy))

                        state_copy = board_copy.copy()
                        state_copy[move_made[0]][move_made[1]] = 0

                        states.append(state_copy)

                        board_copy = np.rot90(board_copy)

                    # Reflection because reflectional symmetry
                    board_copy = board_copy.T

                # Swap symbols because symbolically (almost) symmetric
                gm.swap_symbols(board_copy)

            gm.select_space(board, player, (action[0], action[1]))

        states = np.array(states)
        labels = np.array(labels)

        data_set, label_set = attach_past(states, labels)

        move_count = int(len(data_set)/16)

        for i in range(16):
            augmented_data[i] = data_set[i*move_count:(i+1)*move_count]
            augmented_labels[i] = label_set[i*move_count:(i+1)*move_count]

    return np.array(augmented_data), np.array(augmented_labels)


def attach_past(augmented_history, labels):

    data_set = []
    labels_set = []

    x_turn = np.ones((1, gm.board_len, gm.board_len))
    o_turn = np.zeros((1, gm.board_len, gm.board_len))

    move_count = int(len(augmented_history)/16)

    for i in range(16):

        # empty_x = deque([gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board()], maxlen=5)
        # empty_o = deque([gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board(), gm.create_board()], maxlen=5)

        if i > 7:
            swapped = True
        else:
            swapped = False

        for j in range(move_count):
            state = augmented_history[16*j + i]
            label = labels[16*j + i]

            split = split_state(state)

            # empty_x.appendleft(split[1])
            # empty_o.appendleft(split[0])

            # np_empty_x = np.array(empty_x)
            # np_empty_o = np.array(empty_o)

            if j % 2 == 0:
                if swapped is False:
                    turn = x_turn
                else:
                    turn = o_turn
            else:
                if swapped is False:
                    turn = o_turn
                else:
                    turn = x_turn

            # data_instance = np.r_[np_empty_x, np_empty_o, turn]
            data_instance = np.r_[split, turn]

            # Model-specific reshaping:
            # data_instance = data_instance.reshape((gm.board_len, gm.board_len, 11))
            data_instance = data_instance.reshape((gm.board_len, gm.board_len, 3))
            data_instance = np.moveaxis(data_instance, -1, 0)

            data_set.append(data_instance)
            labels_set.append(label)

    return np.array(data_set), np.array(labels_set)


def translate_cluster(game_hist):
    possible_translations = [game_hist]

    game_copy_y = game_hist.copy()
    in_bounds_y = True

    while in_bounds_y:

        game_copy_x = game_copy_y.copy()
        in_bounds_x = True

        while in_bounds_x:
            for i, move in enumerate(game_copy_x):
                if move[1] > 0:
                    game_copy_x[i][1] -= 1
                else:
                    in_bounds_x = False
                    break

            if in_bounds_x:
                possible_translations.append(game_copy_x.copy())

        game_copy_x = game_copy_y.copy()
        in_bounds_x = True

        while in_bounds_x:
            for i, move in enumerate(game_copy_x):
                if move[1] < 29:
                    game_copy_x[i][1] += 1
                else:
                    in_bounds_x = False
                    break

            if in_bounds_x:
                possible_translations.append(game_copy_x.copy())

        for i, move in enumerate(game_copy_y):
            if move[0] > 0:
                game_copy_y[i][0] -= 1
            else:
                in_bounds_y = False
                break

    game_copy_y = game_hist.copy()
    in_bounds_y = True

    while in_bounds_y:

        game_copy_x = game_copy_y.copy()
        in_bounds_x = True

        while in_bounds_x:
            for i, move in enumerate(game_copy_x):
                if move[1] > 0:
                    game_copy_x[i][1] -= 1
                else:
                    in_bounds_x = False
                    break

            if in_bounds_x:
                possible_translations.append(game_copy_x.copy())

        game_copy_x = game_copy_y.copy()
        in_bounds_x = True

        while in_bounds_x:
            for i, move in enumerate(game_copy_x):
                if move[1] < 29:
                    game_copy_x[i][1] += 1
                else:
                    in_bounds_x = False
                    break

            if in_bounds_x:
                possible_translations.append(game_copy_x.copy())

        for i, move in enumerate(game_copy_y):
            if move[0] < 29:
                game_copy_y[i][0] += 1
            else:
                in_bounds_y = False
                break

    return possible_translations


def discount_win(winner, game_len, discount_rate):
    if winner == 'X':
        x_win_param = 1
        o_win_param = -1
    else:
        x_win_param = -1
        o_win_param = 1

    X_discounted_win = []
    O_discounted_win = []

    for i in range(game_len):
        X_discounted_win.append(x_win_param*discount_rate**(game_len-i-1))
        O_discounted_win.append(o_win_param*discount_rate**(game_len-i-1))

    return X_discounted_win, O_discounted_win


def download_data(game_start_nr=0, game_count=180):

    data_path = "D:/machine_learning_data/TTT/data_flat"

    for i in range(game_count):
        game_nr = game_start_nr + i
        os.mkdir(f"{data_path}/game_{game_nr}")
        game_hists = pull_game_data(game_nr, move_cluster=True)
        for j in range(len(game_hists)):
            os.mkdir(f"{data_path}/game_{game_nr}/perm_{j}")
            data, labels = augment_game([game_hists[j]])
            for k in range(16):
                np.savez_compressed(f"{data_path}/game_{game_nr}/perm_{j}/var_{k}", data=data[k], label=labels[k])


def add_noise_to_labels(labels, mean, std):
    for label in labels:
        ind = np.argmax(label)
        dl_0 = np.clip(np.random.normal(mean, std, label.shape), 0, 1)
        label += dl_0
        dl_1 = np.clip(np.random.normal(1-mean, std, 1), 0, 1)
        label[ind] = dl_1[0]
    return labels


def get_symbol_token(symbol):
    if symbol == 'X':
        return np.ones((1, gm.board_len, gm.board_len))
    else:
        return np.zeros((1, gm.board_len, gm.board_len))


def get_prediction_format(obs, symbol):
    token = get_symbol_token(symbol)
    split = split_state(obs)

    data = np.r_[split, token]

    data = np.moveaxis(data, 0, -1)
    data = np.array([data])

    return data
