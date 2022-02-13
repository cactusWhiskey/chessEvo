import random
import zarr
from collections import deque
import numpy as np
import tensorflow as tf
from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import ChessAgentPolicy
import ChessDataGenerator
import ModifiedPyDriver
from ChessEnv import ChessEnv



class ChessAgent:
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec):
        # hyperparams
        self.hyperparams = {}
        self._build_hyperparams()

        # Counters
        self.target_update_counter = 0

        # Specs
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec

        # Models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.supervised_model = self.build_model()
        self.supervised_model.set_weights(self.model.get_weights())

        # Buffers
        self.replay_buffer = deque(maxlen=self.hyperparams["replay_buffer_size"])

        # Policies
        self.eval_policy = ChessAgentPolicy.ChessAgentPolicy(time_step_spec, action_spec,
                                                             self.model, epsilon_greedy=False)
        self.collect_policy = ChessAgentPolicy.ChessAgentPolicy(time_step_spec, action_spec,
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

    def collect_initial_data(self, env: ChessEnv, max_steps=None, max_episodes=None):
        random_policy = random_py_policy.RandomPyPolicy(self.time_step_spec, self.action_spec)
        driver = ModifiedPyDriver.ModifiedPyDriver(env, random_policy, observers=None,
                                                   transition_observers=[self.replay_buffer.append],
                                                   max_steps=max_steps, max_episodes=max_episodes)
        t_step = env.reset()
        driver.run(t_step)

    def train(self, env: ChessEnv, iterations, max_episodes=None, play_test_every=None):
        if self.model.loss.name == 'sparse_categorical_crossentropy':
            opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
            loss = tf.keras.losses.MeanSquaredError()
            self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

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

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 13)),
            # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(4096)
        ])

        opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        return model

    def test_play(self, env: ChessEnv):
        transitions = []
        env.print = True
        t_step = env.reset()
        driver = ModifiedPyDriver.ModifiedPyDriver(env, self.eval_policy, observers=None,
                                                   transition_observers=[transitions.append],
                                                   max_steps=None, max_episodes=1)
        driver.run(t_step)
        print(transitions[-1][2].reward)
        self.current_time_step = env.reset()
        env.print = False
        return transitions

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        print("Updated Target Model")

    def _build_hyperparams(self):
        self.hyperparams["replay_buffer_size"] = 50000
        self.hyperparams["train_batch_size"] = 32
        self.hyperparams["update_target_every"] = 20
        self.hyperparams["learning_rate"] = 1e-3
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
        self.eval_policy = ChessAgentPolicy.ChessAgentPolicy(self.time_step_spec, self.action_spec,
                                                             self.model, epsilon_greedy=False)
        self.collect_policy = ChessAgentPolicy.ChessAgentPolicy(self.time_step_spec, self.action_spec,
                                                                self.model, epsilon_greedy=True)

        # TimeSteps
        self.current_time_step = None

    def supervised_train(self, train_inputs: np.ndarray, train_labels: np.ndarray,
                         test_inputs: np.ndarray, test_labels: np.ndarray, eps: int):

        if self.model.loss.name == 'mean_squared_error':
            opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        self.model.fit(train_inputs, train_labels, epochs=eps,
                       validation_data=(test_inputs, test_labels))

    def supervised_train_zarr(self, x_train: zarr.Array, y_train: zarr.array,
                              x_test: zarr.array, y_test: zarr.Array,
                              eps=1, num_to_load=100_000, mini_batch_size=32):

        train_gen = ChessDataGenerator.ChessDataGenerator(x_train, y_train, mini_batch_size,
                                                          num_to_load=num_to_load)
        test_gen = ChessDataGenerator.ChessDataGenerator(x_test, y_test, mini_batch_size,
                                                         num_to_load=num_to_load)

        if self.model.loss.name == 'mean_squared_error':
            opt = tf.keras.optimizers.Adam(self.hyperparams["learning_rate"])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        self.model.fit(x=train_gen, validation_data=test_gen, shuffle=False, epochs=eps)
