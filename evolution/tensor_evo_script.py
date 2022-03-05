from evolution import Tensor_Evolution
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

data = x_train, y_train, x_test, y_test
worker = Tensor_Evolution.EvolutionWorker()
worker.evolve(data)
