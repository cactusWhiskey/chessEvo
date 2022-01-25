import random
import matplotlib.pyplot as plt
import numpy
from deap import base
from deap import creator
from deap import tools
import ChessNetwork
import ChessWorker

CX, MUT = 0.5, 0.2
POP_SIZE, T_SIZE, NGEN = 100, 3, 50
MUT_LAYER_PROB = 0.1


def cxUniformShallow(ind1, ind2, layer_prob):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped according to the
    *indpb* probability.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param layer_prob: Probability of crossover happening on a given layer
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    num_layers = min(len(ind1), len(ind2))
    for i in range(num_layers):  # iterate through list of layer weights
        if random.random() < layer_prob:  # select this layer
            # Note: ind[i] is a list of ndarrays
            ind1[i], ind2[i] = ind2[i], ind1[i]  # just swap lists of layer matrices wholesale

    return ind1, ind2


# def cxUniformDeep(ind1, ind2, indpb, layer_prob):
#     """Executes a uniform crossover that modify in place the two
#     :term:`sequence` individuals. The attributes are swapped according to the
#     *indpb* probability.
#     :param ind1: The first individual participating in the crossover.
#     :param ind2: The second individual participating in the crossover.
#     :param indpb: Independent probability for each attribute to be exchanged within a layer.
#     :param layer_prob: Probability of crossover happening on a given layer
#     :returns: A tuple of two individuals.
#     This function uses the :func:`~random.random` function from the python base
#     :mod:`random` module.
#     """
#     numLayers = min(len(ind1), len(ind2))
#
#     for i in range(numLayers):  # iterate through list of layer weights
#         if random.random() < layer_prob:  # select this layer
#             # Note: ind[i] is a list of ndarrays
#
#             matricesInLayer = min(len(ind1[i]), len(ind2[i]))
#             for j in range(matricesInLayer):
#                 # Note ind[i][j] is a single ndarray
#
#                 r,c = ind1[i][j].shape
#                 for row in r:
#                     for col in c:
#                         ind1[i][j][row,col] , ind2[i][j][row,col] = ind2[i][j][row,col] , ind1[i][j][row,col] =
#
#
#
#     return ind1, ind2


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    #  Individual Structure:
    #       fitness = float
    #       list[
    #               0 = list[ndarrays] <- layers
    #                      0 = ndarray (matrix of weights)
    #                      1 = ndarray (typically a bias matrix)
    #                       ...
    #               1 = list[ndarrays]
    #               ...


def build_individual():
    ind = creator.Individual()
    for x in range(5):
        ind.append([random.random() for x in range(5)])
    return ind


def build_individual1():
    model = ChessNetwork.build_model()
    ind = creator.Individual()
    for layer in model.layers[1:]:
        ind.append(layer.get_weights())
    return ind


def evaluate(actor: ChessWorker.ChessWorker, individual):
    actor.reset_board.remote()
    network = ChessNetwork.ChessNetwork()
    network.build_from_genome(individual)
    return actor.play.remote(network)


def evaluate_testing(individual):
    # test eval funtion that just returns sum of the network weights
    total = 0.0
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            total += numpy.sum(individual[i][j])
    return total,


class EvolutionWorker:
    def __init__(self):
        self.record = None
        self.logbook = None
        self.stats = None
        self.pop = None
        setup_creator()
        self.toolbox = base.Toolbox()
        self.setup_toolbox()
        self.setup_pop()
        self.setup_stats()
        self.setup_log()

    def setup_toolbox(self):
        # self.toolbox.register("attribute", build_individual)
        # self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=4)
        self.toolbox.register("individual", build_individual1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", cxUniformShallow, layer_prob=0.1)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("mutate_wrapper", self.mutate_wrapper)
        self.toolbox.register("select", tools.selTournament, tournsize=T_SIZE)

    def setup_pop(self):
        self.pop = self.toolbox.population(n=POP_SIZE)

    def setup_stats(self):
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

    def setup_log(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "max", "min"

    def plot(self):
        gen, avg = self.logbook.select("gen"), self.logbook.select("avg")
        fig, ax = plt.subplots()
        ax.plot(gen, avg, "g-", label="Avg Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness", color="b")
        plt.show()

    def evolve(self):
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUT:
                    self.toolbox.mutate_wrapper(mutant)
                    # self.toolbox.mutate(mutant)
                    # del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring
            self.record = self.stats.compile(self.pop)
            self.logbook.record(gen=g, **self.record)
            print(self.logbook.stream)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s" % best_ind.fitness.values)

        self.plot()

    def mutate_wrapper(self, mutant):
        for i in range(len(mutant)):
            if random.random() < MUT_LAYER_PROB:
                # mutate this layer's matrices
                # mutant[i] is a list of ndarrays
                num_matrices = len(mutant[i])
                for j in range(num_matrices):
                    matrix = mutant[i][j]  # type: numpy.ndarray
                    shape = matrix.shape
                    matrix = matrix.flatten()
                    self.toolbox.mutate(matrix)
                    matrix.reshape(shape)
                    mutant[i][j] = matrix

        del mutant.fitness.values
        return mutant
