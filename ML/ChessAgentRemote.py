
import ray
import ChessAgent
from ChessEnv import ChessEnv

hyperparam_ranges = [
    [400, 800, 1600, 3200, 6400, 10000, 20000, 50000],  # replay buffer size
    [16, 32, 64, 128],  # batch size
    [2, 5, 10, 20, 40, 80, 100],  # update target every
    [0.1, 0.01, 0.001, 0.0001],  # learning rate
    [1]  # collect steps per iteration
]


@ray.remote(num_cpus=1)
class ChessAgentRemote(ChessAgent.ChessAgent):
    def __init__(self):
        #  Remote Agent carries its own env
        self.env = ChessEnv()
        super().__init__(self.env.time_step_spec(), self.env.action_spec())

    def collect_initial_data_remote(self, max_steps=None, max_episodes=None):
        self.collect_initial_data(self.env, max_steps, max_episodes)

    def train_remote(self, iterations, max_episodes=None, play_test_every=None):
        self.train(self.env, iterations, max_episodes, play_test_every)

    def eval_self(self, train_iterations, hyperparam_choices: list):
        i = 0
        for key in self.hyperparams:
            self.hyperparams[key] = hyperparam_ranges[i][hyperparam_choices[i]]
            i += 1
        collect_size = int(self.hyperparams["replay_buffer_size"] * 0.6)
        self.collect_initial_data_remote(max_episodes=collect_size)
        self.train_remote(train_iterations)
        return self.train_metrics[2].result(),

    def close_engine(self):
        self.env.close()




