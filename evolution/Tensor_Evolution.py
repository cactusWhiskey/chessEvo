import random
import numpy
import tensorflow as tf
from deap import creator, base, tools
from matplotlib import pyplot as plt

from evolution import tensor_node, tensor_network

# constants
INPUT_SHAPES = [(28, 28)]
NUM_OUTPUTS = [10]
CX, M_Insert, M_Del, M_Mut = 0.4, 0.3, 0.1, 0.2
POP_SIZE, T_SIZE, NGEN = 10, 3, 30
COMPLEXITY_PENALTY = 0.01
opt = tf.keras.optimizers.Adam(0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
fit_epochs = 5
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.01)


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def initialize_ind(input_shapes, num_outputs) -> list:
    ind = creator.Individual()
    hyper_params = []
    tn = tensor_network.TensorNetwork(input_shapes, num_outputs)
    ind.append(hyper_params)
    ind.append(tn)
    return ind


def build_individual(individual: list):
    tn = individual[1]
    inputs = tn.all_nodes[tn.input_ids[0]](tn.all_nodes, tn.graph)
    outputs = tn.all_nodes[tn.output_ids[0]](tn.all_nodes, tn.graph)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
    return model


def evaluate(individual: list, data: tuple):
    x_train, y_train, x_test, y_test = data
    model = build_individual(individual)
    model.fit(x_train, y_train, epochs=fit_epochs, callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(x_test, y_test)

    length = len(individual[1].get_middle_nodes())
    penalty = COMPLEXITY_PENALTY * length
    return test_acc - penalty,


def mutate_insert(individual: list):
    tn = individual[1]
    position = random.randint(0, len(tn.non_input_nodes) - 1)
    node_type = random.choice(tensor_node.valid_node_types)
    node = tensor_node.create(node_type)
    tn.insert_node(node, position)
    return individual,


def mutate_mutate(individual: list):
    tn = individual[1]
    length = len(tn.get_mutatable_nodes())
    position = random.randint(0, length - 1)
    tn.mutate_node(position)
    return individual,


def mutate_delete(individual: list):
    tn = individual[1]
    length = len(tn.get_middle_nodes())

    if length == 0:  # nothing to delete
        return individual,

    position = random.randint(0, length - 1)
    tn.delete_node(position)
    return individual,


def cx_single_node(ind1, ind2):
    tn = ind1[1]
    other_tn = ind2[1]
    tensor_network.cx_single_node(tn, other_tn)
    return ind1, ind2


def cx_chain(ind1, ind2):
    tn = ind1[1]
    other_tn = ind2[1]
    tensor_network.cx_chain(tn, other_tn)
    return ind1, ind2


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
        self.toolbox.register("individual", initialize_ind, input_shapes=INPUT_SHAPES, num_outputs=NUM_OUTPUTS)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", cx_single_node)
        self.toolbox.register("mutate_insert", mutate_insert)
        self.toolbox.register("mutate_delete", mutate_delete)
        self.toolbox.register("mutate_mutate", mutate_mutate)
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

    def evolve(self, data):
        self.toolbox.register("evaluate", evaluate, data=data)

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
                if random.random() < M_Insert:
                    self.toolbox.mutate_insert(mutant)
                    del mutant.fitness.values

                if random.random() < M_Del:
                    self.toolbox.mutate_delete(mutant)
                    del mutant.fitness.values

                if random.random() < M_Mut:
                    self.toolbox.mutate_mutate(mutant)
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
        build_individual(best_ind).summary()

        # self.plot()
