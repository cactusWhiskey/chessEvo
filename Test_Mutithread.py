import ray
import random

#from ray.util import ActorPool

import ChessWorker
from ActorPoolExtension import ActorPool
import time
import numpy as np


@ray.remote
class Actor:
    def double(self, n):
        return n * 2


def work2(actor: Actor, value: int):
    if value == 2:
        time.sleep(10)
    return actor.double.remote(value)


def work(actor: ChessWorker.ChessWorker, value):
    #print(str(value))
    return actor.play_test.remote(value)

#
# a1, a2 = Actor.remote(), Actor.remote()
# pool = ActorPool([a1, a2])
# l = [x for x in range(10)]
# print(list(pool.map(work2, l)))
# gen = pool.map_unordered(lambda a, v: a.double.remote(v), [1, 2, 3, 4])
# print([v for v in gen])
# #print(pool.map_ordered_return_all(work2, l))

c1, c2 = ChessWorker.ChessWorker.remote(), ChessWorker.ChessWorker.remote()

pool = ActorPool([c1, c2])
print(list(pool.map(work, [1, 2, 3, 4])))
c1.reset_board.remote()
c2.reset_board.remote()
print(list(pool.map_unordered(work, [1, 2, 3, 4])))
c1.reset_board.remote()
c2.reset_board.remote()
print(pool.map_ordered_return_all(work, [1, 2, 3, 4]))
c1.close_engine.remote()
c2.close_engine.remote()
c1.__ray_terminate__.remote()
c2.__ray_terminate__.remote()
