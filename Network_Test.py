import chess
import ray
import ChessWorker
from ChessNetwork import ChessNetwork

net = ChessNetwork()
net.build_model()

worker = ChessWorker.ChessWorker.remote()
id1 = worker.play_test.remote()
ray.wait([id1], num_returns=1)
print(ray.get(id1))
worker.reset_board.remote()
id1 = worker.play.remote(net)
ray.wait([id1], num_returns=1)
print(ray.get(id1))