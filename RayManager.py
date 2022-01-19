from typing import Iterable

import ray

import ChessWorker


class RayManager:
    def __init__(self, numActors:int):
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
