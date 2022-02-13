from collections.abc import Generator
import numpy as np
import zarr


class ChessDataGenerator(Generator):

    def __init__(self, inputs: zarr.Array, labels: zarr.Array, minibatch_size=32,
                 num_to_load=100_000):

        if len(inputs) != len(labels):
            raise ValueError("length inputs != length labels")

        self.minibatch_size = minibatch_size
        self.inputs = inputs
        self.labels = labels
        self.num_to_load = num_to_load
        self.loaded_inputs = []
        self.loaded_labels = []
        self.num_train_examples = None
        self.num_test_examples = None
        self.current_index = 0
        self.load_index = 0
        self.done = False
        self.first = True
        self.rng = np.random.default_rng()

        if (self.num_to_load % self.minibatch_size) != 0:
            extra = self.minibatch_size - (self.num_to_load % self.minibatch_size)
            self.num_to_load += extra

    def send(self, value):
        if self.first:
            self._load_data()
            self.first = False

        #  if current_index outside range of valid indexes
        if self.current_index > (len(self.loaded_inputs) - 1):
            if self.done:
                # nothing left to load in
                raise StopIteration
            else:
                self.current_index = 0
                # load fresh data
                self._load_data()

        return_value = (self.loaded_inputs[self.current_index: self.current_index + self.minibatch_size],
                        self.loaded_labels[self.current_index: self.current_index + self.minibatch_size])

        self.current_index += self.minibatch_size
        return return_value

    def throw(self, typ=None, val=None, tb=None):
        raise StopIteration

    def _load_data(self):
        remaining_records = len(self.labels) - self.load_index

        self.num_to_load = min(self.num_to_load, remaining_records)

        self.loaded_inputs = self.inputs[self.load_index:self.load_index + self.num_to_load]
        self.loaded_labels = self.labels[self.load_index:self.load_index + self.num_to_load]

        self.load_index += self.num_to_load

        if self.load_index > len(self.labels) - 1:
            self.done = True

        rng = np.random.default_rng()
        state = rng.__getstate__()
        rng.shuffle(self.loaded_inputs)
        rng.__setstate__(state)
        rng.shuffle(self.loaded_labels)

    def print_diag(self):
        print("Current Index: " + str(self.current_index) + "    Load Index: " + str(self.load_index)
              + "    Num_to_load: " + str(self.num_to_load) +
              "    Num loaded: " + str(len(self.loaded_inputs)))


def self_test():
    x = np.arange(100)
    y = np.arange(100)
    zx = zarr.array(x)
    zy = zarr.array(y)

    gen = ChessDataGenerator(zx, zy, minibatch_size=3, num_to_load=10)
    gen.print_diag()

    for batch in gen:
        gen.print_diag()
        x, y = batch
        print(str(x))
        print(str(y))
        gen.print_diag()
