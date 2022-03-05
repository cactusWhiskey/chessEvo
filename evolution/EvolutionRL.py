import random
# import matplotlib.pyplot as plt
import numpy
from deap import base
from deap import creator
from deap import tools
from ML import ChessEnv, ChessAgent

# constants
CX, MUT = 0.5, 0.2
POP_SIZE, T_SIZE, NGEN = 25, 3, 30
train_iterations = 10

#  Hyperparm ranges
hyperparams = [
    [400, 800, 1600, 3200, 6400, 10000, 20000, 50000],  # replay buffer size
    [16, 32, 64, 128],  # batch size
    [2, 5, 10, 20, 40, 80, 100],  # update target every
    [0.1, 0.01, 0.001, 0.0001],  # learning rate
    [1]  # collect steps per iteration
]


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual: list, env: ChessEnv):
    agent = ChessAgent.ChessAgent(env.time_step_spec(), env.action_spec())
    i = 0
    for key in agent.hyperparams:
        agent.hyperparams[key] = hyperparams[i][individual[i]]
        i += 1
    # agent.hyperparams["replay_buffer_size"] = hyperparams[0][individual[0]]
    # agent.hyperparams["train_batch_size"] = hyperparams[1][individual[1]]
    # agent.hyperparams["update_target_every"] = hyperparams[2][individual[2]]
    # agent.hyperparams["learning_rate"] = hyperparams[3][individual[3]]

    collect_size = int(agent.hyperparams["replay_buffer_size"] * 0.6)
    agent.collect_initial_data(env, max_episodes=collect_size)
    agent.train(env, train_iterations)
    return agent.train_metrics[2].result(),


def build_individual() -> list:
    ind = creator.Individual()
    for param in hyperparams:
        ind.append(random.randint(0, len(param) - 1))
    return ind


def mutate(individual: list, indpb):
    for index, attr in enumerate(individual):
        if random.random() < indpb:
            individual[index] = random.randint(0, len(hyperparams[index]) - 1)
    return individual


class EvolutionWorker:
    def __init__(self):
        self.env = ChessEnv.ChessEnv()
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
        self.toolbox.register("evaluate", evaluate, env=self.env)
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
            fitnesses = list(map(self.toolbox.evaluate, self.pop))
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
        print("Replay Buffer Size: " + str(hyperparams[0][best_ind[0]]))
        print("Batch Size: " + str(hyperparams[1][best_ind[1]]))
        print("Update Target Every: " + str(hyperparams[2][best_ind[2]]))
        print("Learning Rate: " + str(hyperparams[3][best_ind[3]]))
        self.env.close()
        # self.plot()
