import random
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import tensorflow as tf
from tf_agents.environments import PyEnvironment
from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from ML.DqnAgentPolicy import DqnAgentPolicy
from ML.ModifiedPyDriver import ModifiedPyDriver


class DqnAgent(ABC):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec,
                 replay_buffer_size=50_000):
        # hyperparams
        self.hyperparams = {}
        self._build_hyperparams(replay_buffer_size)

        # Counters
        self.target_update_counter = 0

        # Specs
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec

        # Models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        # Buffers
        self.replay_buffer = deque(maxlen=self.hyperparams["replay_buffer_size"])

        # Policies
        self.eval_policy = DqnAgentPolicy(time_step_spec, action_spec,
                                          self.model, epsilon_greedy=False)
        self.collect_policy = DqnAgentPolicy(time_step_spec, action_spec,
                                             self.model, epsilon_greedy=True)

        # TimeSteps
        self.current_time_step = None

        # Metrics
        self.train_metrics = [
            py_metrics.NumberOfEpisodes(),
            py_metrics.EnvironmentSteps(),
            py_metrics.AverageReturnMetric(),
            py_metrics.AverageEpisodeLengthMetric(),
        ]

    def collect_initial_data(self, env: PyEnvironment, max_steps=None, max_episodes=None):
        random_policy = random_py_policy.RandomPyPolicy(self.time_step_spec, self.action_spec)
        driver = ModifiedPyDriver.ModifiedPyDriver(env, random_policy, observers=None,
                                                   transition_observers=[self.replay_buffer.append],
                                                   max_steps=max_steps, max_episodes=max_episodes)
        t_step = env.reset()
        driver.run(t_step)

    def train(self, env: PyEnvironment, iterations: int, max_episodes=None, play_test_every=None):

        play_test_counter = 0
        self.current_time_step = env.reset()
        driver = ModifiedPyDriver.ModifiedPyDriver(env, self.collect_policy, observers=self.train_metrics,
                                                   transition_observers=[self.replay_buffer.append],
                                                   max_steps=self.hyperparams["collect_steps_per_iter"],
                                                   max_episodes=max_episodes)

        for _ in range(iterations):
            self.current_time_step, _ = driver.run(self.current_time_step)

            batch = random.sample(self.replay_buffer, self.hyperparams["train_batch_size"])

            current_observations = np.array([transition[0].observation for transition in batch])
            current_Qs = self.model.predict(current_observations)

            next_observations = np.array([transition[2].observation for transition in batch])
            future_Qs = self.target_model.predict(next_observations)

            X = []
            Y = []
            for index, transition in enumerate(batch):
                max_future_Q = np.max(future_Qs[index])
                new_Q = transition[2].reward + transition[2].discount * max_future_Q

                action = int(transition[1].action)  # convert ndarray to int
                current_Qs[index][action] = new_Q
                X.append(current_observations[index])  # list of inputs for this current state
                Y.append(current_Qs[index])  # list of outputs for this current state

            self.model.fit(np.array(X), np.array(Y), batch_size=self.hyperparams["train_batch_size"],
                           shuffle=False, verbose=1)

            self.target_update_counter += 1
            play_test_counter += 1

            if self.target_update_counter > self.hyperparams["update_target_every"]:
                self._update_target_model()

            if play_test_every is not None:
                if play_test_counter > play_test_every:
                    self.test_play(env)
                    play_test_counter = 0

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        return None

    def test_play(self, env: PyEnvironment):
        pass

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        print("Updated Target Model")

    def _build_hyperparams(self, replay_buffer_size=50000):
        self.hyperparams["replay_buffer_size"] = replay_buffer_size
        self.hyperparams["train_batch_size"] = 32
        self.hyperparams["update_target_every"] = 20
        self.hyperparams["learning_rate"] = 1e-4
        self.hyperparams["collect_steps_per_iter"] = 2

    def reset_agent(self):
        # Counters
        self.target_update_counter = 0

        # Models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        # Buffers
        self.replay_buffer = deque(maxlen=self.hyperparams["replay_buffer_size"])

        # Policies
        self.eval_policy = DqnAgentPolicy(self.time_step_spec, self.action_spec,
                                          self.model, epsilon_greedy=False)
        self.collect_policy = DqnAgentPolicy(self.time_step_spec, self.action_spec,
                                             self.model, epsilon_greedy=True)

        # TimeSteps
        self.current_time_step = None
