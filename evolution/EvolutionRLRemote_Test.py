import ray
from ML import ChessAgentRemote
import EvolutionRL_Remote
from ActorPoolExtension import ActorPoolExtension

NUM_ACTORS = 4
actors = []
ray.init()

if __name__ == "__main__":
    for x in range(NUM_ACTORS):
        actor = ChessAgentRemote.ChessAgentRemote.remote()
        actors.append(actor)

    pool = ActorPoolExtension(actors)

    evo_worker = EvolutionRL_Remote.EvolutionWorker()

    evo_worker.evolve(pool)

    for actor in actors:
        actor.close_engine.remote()
        actor.__ray_terminate__.remote()
