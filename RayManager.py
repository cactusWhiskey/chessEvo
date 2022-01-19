from typing import Iterable

import ray

import ChessNetwork
import ChessWorker


class RayManager:
    def __init__(self, numActors: int):
        self.index = 0
        self.tasksInProgress = 0
        self.tasksLeft = 0
        self.ids = []
        self.numActors = numActors
        self.actorHandles = []
        self.idMap = {}
        for x in range(numActors):
            self.actorHandles.append(ChessWorker.ChessWorker.remote())

    def executeTasks(self, individuals):
        ready, notReady = ray.wait(list(self.idMap.keys()), num_returns=1)
        print(ray.get(ready[0]))
        actor = self.idMap[ready[0]]
        self.idMap.pop(ready[0])
        net = ChessNetwork.ChessNetwork()
        net.build_from_genome(individuals[self.index])
        self.idMap[actor.play.remote(net)] = actor
        self.tasksLeft -= 1
        self.index += 1

    def executeLastTasks(self):
        ready, notReady = ray.wait(list(self.idMap.keys()), num_returns=self.tasksInProgress)
        print(*ray.get(ready), sep="\n")

    def work(self, individuals: list):
        self.ids = []
        self.index = 0
        self.tasksLeft = len(individuals)
        self.tasksInProgress = 0
        for actor in self.actorHandles:
            net = ChessNetwork.ChessNetwork()
            net.build_from_genome(individuals[self.index])
            self.ids.append(actor.play.remote(net))
            self.index += 1
            self.tasksLeft -= 1
            self.tasksInProgress += 1
        self.idMap = dict(zip(self.ids, self.actorHandles))

        while self.tasksLeft > 0:
            self.executeTasks(individuals)

        self.executeLastTasks()
