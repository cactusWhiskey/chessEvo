import random
# import matplotlib.pyplot as plt
import numpy
from deap import base
from deap import creator
from deap import tools
import ChessAgentRemote
from ActorPoolExtension import ActorPoolExtension

CX, MUT = 0.5, 0.2
POP_SIZE, T_SIZE, NGEN = 20, 3, 30
train_iterations = 50


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(agent: ChessAgentRemote, individual: list):
    param_choices = [x for x in individual]
    return agent.eval_self.remote(train_iterations, param_choices)


def build_individual() -> list:
    ind = creator.Individual()
    for param in ChessAgentRemote.hyperparam_ranges:
        ind.append(random.randint(0, len(param) - 1))
    return ind


def mutate(individual: list, indpb):
    for index, attr in enumerate(individual):
        if random.random() < indpb:
            individual[index] = random.randint(0, len(ChessAgentRemote.hyperparam_ranges[index]) - 1)
    return individual


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
        self.toolbox.register("individual", build_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", mutate, indpb=0.1)
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
        self.logbook.header = "gen", "avg", "max", "min"  # , "select", "cross", "mut", "fit", "log"

    # def plot(self):
    #     gen, avg = self.logbook.select("gen"), self.logbook.select("avg")
    #     fig, ax = plt.subplots()
    #     ax.plot(gen, avg, "g-", label="Avg Fitness")
    #     ax.set_xlabel("Generation")
    #     ax.set_ylabel("Fitness", color="b")
    #     plt.show()

    def evolve(self, pool: ActorPoolExtension):
        # Evaluate the entire population
        fitnesses = pool.map_ordered_return_all(self.toolbox.evaluate, self.pop)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # apply mutation
            for mutant in offspring:
                if random.random() < MUT:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = pool.map_ordered_return_all(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            self.pop[:] = offspring
            self.record = self.stats.compile(self.pop)
            self.logbook.record(gen=g, **self.record)
            print(self.logbook.stream)

            for actor in pool._idle_actors:
                actor.reset_agent.remote()

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s" % best_ind.fitness.values)
        print("Replay Buffer Size: " + str(ChessAgentRemote.hyperparam_ranges[0][best_ind[0]]))
        print("Batch Size: " + str(ChessAgentRemote.hyperparam_ranges[1][best_ind[1]]))
        print("Update Target Every: " + str(ChessAgentRemote.hyperparam_ranges[2][best_ind[2]]))
        print("Learning Rate: " + str(ChessAgentRemote.hyperparam_ranges[3][best_ind[3]]))
        # self.plot()
