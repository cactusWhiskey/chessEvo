import ray
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from evolution import Evolution
from evolution.ActorPoolExtension import ActorPoolExtension
from void import ChessWorker

NUM_ACTORS = 4
actors = []
ray.init()

if __name__ == "__main__":
    for x in range(NUM_ACTORS):
        actor = ChessWorker.ChessWorker.remote()
        actors.append(actor)

    pool = ActorPoolExtension(actors)

    evo_worker = Evolution.EvolutionWorker()

    evo_worker.evolve(pool)

    for actor in actors:
        actor.close_engine.remote()
        actor.__ray_terminate__.remote()
