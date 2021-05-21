# Description:
# - Here the models will play games against each other and learn to play better
# - This is also where the data will be generated and chosen

# Imports:
import numpy as np
import environment as env
import game_mechanics as gm
import model_new as model
import tensorflow as tf
import data_functions as df
from tensorflow.keras.utils import to_categorical
from collections import deque
import random


# Functions:
@tf.function(experimental_relax_shapes=True)
def gradient_mean(layer_var):
    return tf.reduce_mean(layer_var, axis=0)


def append_opponent(op_list, opponent):
    op_list.append(opponent.policy_network.get_weights())
    return


def choose_opponent(opponent_model, list_of_opponents):
    opponent_network = random.choice(list_of_opponents)
    opponent_model.policy_network.set_weights(opponent_network)
    return


def generate_region_mask():
    default_mask = np.zeros((gm.board_len, gm.board_len))

    x_limit = random.randint(0, 20)
    y_limit = random.randint(0, 20)

    x_limit = (x_limit, x_limit+10)
    y_limit = (y_limit, y_limit+10)

    patch = np.ones((10, 10))

    default_mask[y_limit[0]:y_limit[1], x_limit[0]:x_limit[1]] = patch

    return default_mask


def save_gradients(agent, iterator):
    for ind, step_grad in enumerate(agent.gradient_memory):
        for jnd, step_layer in enumerate(step_grad):
            agent.gradient_memory[ind][jnd] = step_layer.numpy()

        np.savez_compressed(gradient_path + f'grad{iterator}', step_gradient=np.array(agent.gradient_memory[ind], dtype=object))
        iterator += 1

    return iterator


# Creation of environment:
ttt_env = env.TTTenvironment()

# Paths:
gradient_path = 'D:/machine_learning_data/TTT/gradients/'

grad_iterator = 0

# Distinction between model-opponent:
learning_rate = 1e-6
weight_decay = 0.0
model_agent = model.AlphaTTT('X', policy_name='policy_net_new', learning_rate=learning_rate, weight_decay=weight_decay)
opponent_agent = model.AlphaTTT('O', policy_name='policy_net_new', learning_rate=learning_rate, weight_decay=weight_decay)

opponent_list = deque([], maxlen=20)
append_opponent(opponent_list, opponent_agent)

# Creation of agents:
X_agent = model_agent
O_agent = opponent_agent

# Training:
# -----------------------------------------------------
# Hyperparameters:
# ================
game_count = 10001

# game_per_render % 2 needs to be 1 so you see agent playing as both X and O
games_per_render = 101
games_per_update = 5
step_batch = 100
games_per_reset_symmetry = 50
initial_epsilon = 0.4
epsilon_decay = 0.9998
discount_rate = 0.9


games_per_save = 500
# This needs to be the same as games_per_reset_symmetry because that will make the ratios more accurate
games_per_update_opponent = 100
model_nr = 1

x_win_count = 0
model_win_count = 0
o_win_count = 0
opponent_win_count = 0

x_getting_crushed = False
o_getting_crushed = False

x_win_list = []
o_win_list = []

for game in range(game_count):
    print(f'Game #{game}')
    winner = None
    move_count = 0
    # Starting player is X
    player = 'X'
    # Choose opponent:
    choose_opponent(opponent_agent, opponent_list)
    first_move = True
    obs, done = ttt_env.reset(), False

    while not done:
        move_count += 1
        # Predicting action

        if player == 'X':
            # X_agent.memorize_state(obs)
            action, was_random = X_agent.policy_predict_action(obs, initial_epsilon, first_move=first_move, model_playing=(model_agent.symbol == 'X'))
            if first_move is True:
                first_move = False
        else:
            # O_agent.memorize_state(obs)
            action, was_random = O_agent.policy_predict_action(obs, initial_epsilon, model_playing=(model_agent.symbol == 'O'))

        # Remember previous observation:
        prev_obs = obs.copy()

        # Save gradients:
        # No teacher learning so opponent agent doesnt calculate gradients
        if player == 'X' and model_agent.symbol == 'X':
            X_agent.compute_policy_gradients(obs, action, memorize=True)
        elif player == 'O' and model_agent.symbol == 'O':
            O_agent.compute_policy_gradients(obs, action, memorize=True)

        # Take the step:
        obs, reward, done, info = ttt_env.step(player, action, was_random)
        winner = info

        # Swap turns
        if player == 'X':
            player = 'O'
        else:
            player = 'X'

    # See if model is getting better
    print(f'Move count: {move_count}, Winner: {winner}, Model played as: {model_agent.symbol}')

    # Decay epsilon:
    initial_epsilon *= epsilon_decay

    if winner == 'X':
        x_win_count += 1
        if model_agent.symbol == 'X':
            model_win_count += 1
        else:
            opponent_win_count += 1
    elif winner == 'O':
        o_win_count += 1
        if model_agent.symbol == 'O':
            model_win_count += 1
        else:
            opponent_win_count += 1

    X_discounted_win, O_discounted_win = df.discount_win(winner, move_count, discount_rate=discount_rate)

    if model_agent.symbol == 'X':
        X_agent.apply_win_to_gradients(X_discounted_win)
    else:
        O_agent.apply_win_to_gradients(O_discounted_win)

    # # Append the game-by-game gradient list
    # Add gradients to file
    if model_agent.symbol == 'X':
        grad_iterator = save_gradients(X_agent, grad_iterator)
    else:
        grad_iterator = save_gradients(O_agent, grad_iterator)

    # Clear last game from memory
    X_agent.clear_memory()
    O_agent.clear_memory()

    # Render game:
    if game % games_per_render == 0 and game != 0:
        ttt_env.render(show_randomness=True)

    x_win_ratio = x_win_count / (x_win_count + o_win_count)
    o_win_ratio = o_win_count / (x_win_count + o_win_count)

    model_win_ratio = model_win_count / (model_win_count + opponent_win_count)
    opponent_win_ratio = opponent_win_count / (model_win_count + opponent_win_count)

    print(f'X/O win ratios of the current 50 games: {x_win_ratio}/{o_win_ratio}')
    print(f'Model/Opponent win ratios of the current 50 games: {model_win_ratio}/{opponent_win_ratio}')

    if x_win_ratio >= 0.6:
        o_getting_crushed = True
    else:
        o_getting_crushed = False

    if o_win_ratio >= 0.6:
        x_getting_crushed = True
    else:
        x_getting_crushed = False

    if game % games_per_update == 0 and game != 0:
        # Update policy
        if model_agent.symbol == 'X':
            X_agent.apply_policy_gradients(step_batch, grad_iterator)
        else:
            O_agent.apply_policy_gradients(step_batch, grad_iterator)

        grad_iterator = 0

    if model_agent.symbol == 'X':
        model_agent.symbol = 'O'
        opponent_agent.symbol = 'X'
        X_agent = opponent_agent
        O_agent = model_agent
    else:
        model_agent.symbol = 'X'
        opponent_agent.symbol = 'O'
        X_agent = model_agent
        O_agent = opponent_agent

    # IMPORTANT:
    # If it's time to swap opponents, add the current policy network to list of past networks and choose one at random
    if game % games_per_update_opponent == 0 and game != 0:
        # Copy over updates to opponent:
        if model_win_ratio > 0.5:
            pass
            # print("Updating opponent...")
            # opponent_agent.policy_network.set_weights(model_agent.policy_network.get_weights())

            # Randomly choose from old opponent and append current network:

        # Regardless of win rate, add to list of opponents
        append_opponent(opponent_list, model_agent)

    if game % games_per_reset_symmetry == 0 and game != 0:
        print("Resetting ratios...")
        x_win_count = 0
        o_win_count = 0
        model_win_count = 0
        opponent_win_count = 0

    if game % games_per_save == 0 and game != 0:
        print("Saving model...")
        agent_name = f'policy_reinforced_mk{model_nr}'
        tf.keras.models.save_model(model_agent.policy_network, './models/' + agent_name, save_format='h5')
        model_nr += 1
    print("==========================")
