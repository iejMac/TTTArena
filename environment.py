# Description:
# The environment class that the agent will interact with

# Imports:
import game_mechanics as gm
import GUI
import numpy as np


# Class:

class TTTenvironment():
    def __init__(self):
        self.board = gm.create_board()
        self.move_memory = []
        self.reward_memory = []
        # Change so it's variable
        self.action_space = self.board.shape
        self.observation_space = self.board.shape
        # Optimization variables:
        # 1. Storing previous evaluation so I don't evaluate each position twice:
        self.evaluation_memory = None

    def reset(self):
        self.board = gm.create_board()
        self.move_memory = []
        self.reward_memory = []
        self.evaluation_memory = None
        obs = self.board
        return obs

    def step(self, player, action, random_action):
        # Remember (player, move):
        self.move_memory.append((player, action, random_action))

        gm.select_space(board=self.board, player=player, position=action)

        # Observation:
        obs = self.board

        # Reward:
        reward = 0

        over = gm.game_over(self.board)

        # Done:
        done = False if over == -1 else True

        winner = None

        if done:
            if over == 2:
                winner = 'X'
            else:
                winner = 'O'

        # Remember (player, reward):
        self.reward_memory.append((player, reward))

        # Info:
        info = winner

        return obs, reward, done, info

    def render(self, speed=500, show_randomness=False):
        # Hard to do this with online stream of moves so after each game render will replay the game with
        # a time delay between moves
        GUI.play_game(mode='render', memory=self.move_memory, speed=speed, show_randomness=show_randomness)
        return


# Functions:

def discount_reward(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_reward(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


