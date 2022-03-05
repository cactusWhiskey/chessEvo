from typing import Iterable

import ray

from void import ChessWorker


class RayManager:
    def __init__(self, numActors):
        self.tasksInProgress = 0
        self.tasksLeft = 0
        self.ids = []
        self.numActors = numActors
        self.actorHandles = []
        self.idMap = {}
        for x in range(numActors):
            self.actorHandles.append(ChessWorker.ChessWorker.remote())

    def executeTask(self):
        ready, notReady = ray.wait(list(self.idMap.keys()), num_returns=1)
        print(ray.get(ready[0]))
        actor = self.idMap[ready[0]]
        self.idMap.pop(ready[0])
        self.idMap[actor.play_test.remote()] = actor
        self.tasksLeft -= 1

    def executeLastTasks(self):
        ready, notReady = ray.wait(list(self.idMap.keys()), num_returns=self.tasksInProgress)
        print(*ray.get(ready), sep = "\n")

    def work(self, values: Iterable):
        self.ids = []
        self.tasksLeft = len(values)
        self.tasksInProgress = 0
        for actor in self.actorHandles:
            self.ids.append(actor.play_test.remote())
            self.tasksLeft -= 1
            self.tasksInProgress += 1
        self.idMap = dict(zip(self.ids, self.actorHandles))

        while self.tasksLeft > 0:
            self.executeTask()

        self.executeLastTasks()


if __name__ == "__main__":
    # c1, c2 = CHessWOrker.chessworker.remote(), CHessWOrker.chessworker.remote()
    #
    # print("c1")
    # id1 = (c1.play.remote())
    # print("c2")
    # id2 = (c2.play.remote())
    #
    # print("wait")
    # ray.wait([id1, id2], num_returns=2)
    # print("get")
    # res1, res2 = ray.get([id1, id2])
    # print(res1)
    # print(res2)
    manager = RayManager(2)
    manager.work([1, 2, 3, 4, 5, 6])
