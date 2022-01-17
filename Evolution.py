import random
from deap import base
from deap import creator
from deap import tools


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


class EvolutionWorker:
    def __init__(self):
        self.toolbox = base.Toolbox()

    def setup_toolbox(self):
        self.toolbox.register("attribute", random.random())
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_bool, 10)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    