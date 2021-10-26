from agent import Agent
from numpy.random import randint

class RandomAgent(Agent):
	def __init__(self, name):
		super().__init__(name)

	def make_action(self, state):
		movex = randint(0, state.shape[1])
		movey = randint(0, state.shape[0])
		return (movey, movex)


