import zarr
import tensorflow as tf
from ML import ChessEnv, ChessAgent

#tf.config.run_functions_eagerly(True)

x_train_file = '/home/ricardo/Downloads/pgn_save/obsZ_train.zarr'
y_train_file = '/home/ricardo/Downloads/pgn_save/movesZ_train.zarr'
x_test_file = '/home/ricardo/Downloads/pgn_save/obsZ_test.zarr'
y_test_file = '/home/ricardo/Downloads/pgn_save/movesZ_test.zarr'

# x_train_file = '/home/arjudoso/bucket/pgn_save/obsZ_train.zarr'
# y_train_file = '/home/arjudoso/bucket/pgn_save/movesZ_train.zarr'
# x_test_file = '/home/arjudoso/bucket/pgn_save/obsZ_test.zarr'
# y_test_file = '/home/arjudoso/bucket/pgn_save/movesZ_test.zarr'

x_train = zarr.open(x_train_file, mode="r")
y_train = zarr.open(y_train_file, mode="r")
x_test = zarr.open(x_test_file, mode="r")
y_test = zarr.open(y_test_file, mode="r")

env = ChessEnv.ChessEnv()
agent = ChessAgent.ChessAgent(env.time_step_spec(), env.action_spec())
agent.hyperparams["learning_rate"] = 1e-4
agent.supervised_train_zarr(x_train, y_train, x_test, y_test, load_checkpoint=False)
