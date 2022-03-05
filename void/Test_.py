import random
import time

from ray.util import ActorPool
import ray


@ray.remote
class Actor:
    def double(self, n):
        return n * 2

def work(a,v):
    return a.double.remote(v)


a1, a2 = Actor.remote(), Actor.remote()
pool = ActorPool([a1, a2])

# pool.map(..) returns a Python generator object ActorPool.map
# gen = pool.map_unordered(lambda a, v: a.double.remote(v), [x for x in range(10)])
gen = pool.map_unordered(work, [x for x in range(10)])
print([v for v in gen])
# [2, 4, 6, 8]
