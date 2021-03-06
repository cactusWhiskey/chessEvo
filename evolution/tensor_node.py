import itertools
import random
import math
from abc import ABC, abstractmethod
import sympy.ntheory as sym
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from networkx import DiGraph

# valid_node_types = ["Conv2D", "Dense", "Flatten", "MaxPooling2D", "ReLU", "BatchNormalization"]
valid_node_types = ["Dense", "ReLU", "BatchNormalization", "MaxPooling2D", "Conv2D"]
dense_max_power_two = 10
max_conv2d_power = 8
conv2d_kernel_sizes = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
max_pooling_size = [(2, 2), (3, 3)]


class TensorNode(ABC):
    id_iter = itertools.count()

    def __init__(self):
        self.id = next(TensorNode.id_iter)
        self.label = self.get_label()
        self.is_branch_root = False
        self.saved_layers = None
        self.can_mutate = False
        self.input_shape = []  # TensorShape (batch, rows, columns, channels)
        self.output_shape = None

    def __eq__(self, other):
        return self.id == other.id

    def __call__(self, all_nodes: dict, graph: DiGraph) -> KerasTensor:
        if self.saved_layers is not None:
            return self.saved_layers
        else:
            self.reset()
            layers_so_far = self._call_parents(all_nodes, graph)

            for parent in self.get_parents(all_nodes, graph):
                self.input_shape.append(parent.output_shape)

            layers_so_far = self.fix_input(layers_so_far)
            layers_so_far = self._build(layers_so_far)
            self.output_shape = layers_so_far.shape

            if self.is_branch_root:
                self.saved_layers = layers_so_far

            return layers_so_far

    def get_parents(self, all_nodes: dict, graph: DiGraph) -> list:
        parents_ids = graph.predecessors(self.id)
        parents = []
        for p_id in parents_ids:
            parents.append(all_nodes[p_id])
        return parents

    # def get_children(self):
    #     return self.graph.successors(self.id)

    def reset(self):
        self.input_shape = []
        self.output_shape = None
        self.saved_layers = None

    @abstractmethod
    def _build(self, layers_so_far) -> KerasTensor:
        raise NotImplementedError

    def _call_parents(self, all_nodes: dict, graph: DiGraph) -> KerasTensor:
        parents = self.get_parents(all_nodes, graph)
        if len(parents) > 0:
            return (parents[0])(all_nodes, graph)
        else:
            return None

    def mutate(self):
        pass

    def fix_input(self, layers_so_far):
        return layers_so_far

    @staticmethod
    @abstractmethod
    def create_random():
        raise NotImplementedError

    def get_label(self) -> str:
        label = str(type(self)).split('.')[-1]
        label = label.split('\'')[0]
        return label


class InputNode(TensorNode):
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape.append(input_shape)
        self.is_branch_root = True

    def _build(self, layers_so_far):
        return tf.keras.Input(shape=self.input_shape[0])

    def reset(self):
        self.output_shape = None
        self.saved_layers = None

    @staticmethod
    def create_random():
        raise NotImplementedError("Input Node can't be created randomly")


class FlattenNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor):
        return tf.keras.layers.Flatten()(layers_so_far)

    @staticmethod
    def create_random():
        return FlattenNode()


class AdditionNode(TensorNode):

    def _call_parents(self, all_nodes: dict, graph: DiGraph) -> tuple:
        parents = self.get_parents(all_nodes, graph)
        return parents[0](all_nodes, graph), parents[1](all_nodes, graph)

    def _build(self, layers_so_far: tuple) -> KerasTensor:
        main_branch, adder_branch = layers_so_far
        return tf.keras.layers.add([main_branch, adder_branch])

    @staticmethod
    def create_random():
        return AdditionNode()

    def fix_input(self, layers_so_far: tuple) -> tuple:
        main_branch, adder_branch = layers_so_far
        main_shape = (self.input_shape[0])
        adder_shape = (self.input_shape[1])

        if main_shape[1:] != adder_shape[1:]:
            adder_branch = tf.keras.layers.Flatten()(adder_branch)
            if main_shape.rank == 2:  # dense shape
                units = main_shape[1]  # main_shape[0] will be None
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)
            elif main_shape.rank == 4:  # Conv2D shape
                units = main_shape[3]  # num of filters
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)
                ones = tf.ones(main_shape[1:])
                tf.expand_dims(ones, 0)
                adder_branch = tf.keras.layers.multiply([ones, adder_branch])
            else:
                main_branch = tf.keras.layers.Flatten()(main_branch)
                main_shape = main_branch.shape
                units = main_shape[1]
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)

        return main_branch, adder_branch


class DenseNode(TensorNode):
    def __init__(self, num_units: int, activation='relu'):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.can_mutate = True

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.Dense(self.num_units,
                                     activation=self.activation)(layers_so_far)

    @staticmethod
    def create_random():
        random_power = random.randint(0, dense_max_power_two)
        units = 2 ** random_power
        return DenseNode(units)

    def mutate(self):
        random_power = random.randint(0, dense_max_power_two)
        units = 2 ** random_power
        self.num_units = units


class Conv2dNode(TensorNode):
    def __init__(self, filters, kernel_size, padding='same'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = 'relu'
        self.padding = padding
        self.can_mutate = True

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        print("Shape (build): " + str(layers_so_far.shape))
        return tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                      activation=self.activation,
                                      padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        random_power = random.randint(0, max_conv2d_power)
        filters = 2 ** random_power
        kernel_size = random.choice(conv2d_kernel_sizes)
        return Conv2dNode(filters, kernel_size)

    def mutate(self):
        random_power = random.randint(0, max_conv2d_power)
        filters = 2 ** random_power
        kernel_size = random.choice(conv2d_kernel_sizes)

        self.filters = filters
        self.kernel_size = kernel_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        return reshape_1D_to_2D(layers_so_far)


class ReluNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.ReLU()(layers_so_far)

    @staticmethod
    def create_random():
        return ReluNode()


class BatchNormNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.BatchNormalization()(layers_so_far)

    @staticmethod
    def create_random():
        return BatchNormNode()


class OutputNode(TensorNode):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def _build(self, layers_so_far) -> KerasTensor:
        layers_so_far = tf.keras.layers.Flatten()(layers_so_far)
        layers_so_far = tf.keras.layers.Dense(self.num_outputs, activation=None)(layers_so_far)
        return layers_so_far

    @staticmethod
    def create_random():
        raise NotImplementedError("Output Node can't be created randomly")


class MaxPool2DNode(TensorNode):
    def __init__(self, pool_size=(2, 2), padding="valid"):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding
        self.can_mutate = True

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        print("Shape (build): " + str(layers_so_far.shape))
        return tf.keras.layers.MaxPooling2D(self.pool_size,
                                            padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        pool_size = random.choice(max_pooling_size)
        return MaxPool2DNode(pool_size=pool_size)

    def mutate(self):
        pool_size = random.choice(max_pooling_size)
        self.pool_size = pool_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        return reshape_1D_to_2D(layers_so_far)


def create(node_type: str) -> TensorNode:
    if node_type == "Conv2D":
        return Conv2dNode.create_random()
    elif node_type == "Dense":
        return DenseNode.create_random()
    elif node_type == "Flatten":
        return FlattenNode()
    elif node_type == "MaxPooling2D":
        return MaxPool2DNode.create_random()
    elif node_type == "ReLU":
        return ReluNode.create_random()
    elif node_type == "add":
        return AdditionNode.create_random()
    elif node_type == "BatchNormalization":
        return BatchNormNode.create_random()
    else:
        raise ValueError("Unsupported node type: " + str(node_type))


def evaluate_square_shapes(n) -> tuple:
    sqrt = int(math.sqrt(n)) - 1

    while sqrt >= 5:
        square = sqrt ** 2
        channels = n // square

        if (square * channels) == n:
            return sqrt, sqrt, channels

        sqrt -= 1
    return None


def is_square_rgb(n):
    if (n % 3) == 0:
        n = n // 3
        return sym.primetest.is_square(n)
    else:
        return False


def shape_from_primes(n) -> tuple:
    prime_dict = sym.factorint(n)
    prime_list = []

    for prime, repeat in prime_dict.items():
        for rep in range(repeat):
            prime_list.append(prime)

    if len(prime_list) == 2:
        return prime_list[0], prime_list[1], 1

    while len(prime_list) > 3:
        prime_list.sort(reverse=True)
        composite = prime_list[-1] * prime_list[-2]
        prime_list.pop(-1)
        prime_list.pop(-1)
        prime_list.append(composite)

    return prime_list[0], prime_list[1], prime_list[2]


def reshape_1D_to_2D(layers_so_far: KerasTensor) -> KerasTensor:
    if layers_so_far.shape.rank != 2:
        raise ValueError("reshape_1D_to_2D only accepts KerasTensors of rank 2  (batch_size, dim1)")

    n = layers_so_far.shape[1]

    if sym.isprime(n):
        target_shape = (1, 1, n)

    elif sym.primetest.is_square(n):
        sqrt = int(math.sqrt(n))
        target_shape = (sqrt, sqrt, 1)

    elif is_square_rgb(n):
        n = n / 3
        sqrt = int(math.sqrt(n))
        target_shape = (sqrt, sqrt, 3)

    else:
        target_shape = evaluate_square_shapes(n)
        if target_shape is None:
            target_shape = shape_from_primes(n)

    return tf.keras.layers.Reshape(target_shape)(layers_so_far)
