import ray
from void import ChessWorker
from void.ChessNetwork import ChessNetwork

net = ChessNetwork()


worker = ChessWorker.ChessWorker.remote()
# id1 = worker.play_test.remote()
# ray.wait([id1], num_returns=1)
# print(ray.get(id1))
# worker.reset_board.remote()
id1 = worker.play.remote(net)
ray.wait([id1], num_returns=1)
print(ray.get(id1))
