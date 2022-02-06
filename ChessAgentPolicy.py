import random

import numpy as np
from typing import Optional
from keras import Model
from tf_agents.policies import py_policy, random_py_policy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types


class ChessAgentPolicy(py_policy.PyPolicy):

    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec, model: Model,
                 epsilon=1.0, epsilon_greedy=True, decay=.9995, min_epsilon=0.3):
        super().__init__(time_step_spec, action_spec)
        self.model = model
        self.epsilon = epsilon
        self.epsilon_greedy = epsilon_greedy
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.random_policy = random_py_policy.RandomPyPolicy(self.time_step_spec, self.action_spec)

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:

        if self.epsilon_greedy:
            if random.random() > self.epsilon:
                output = self.model.predict(np.expand_dims(time_step.observation, 0))  # ndarray shape (1,4096)
                max_logit_index = np.argmax(output[0])
                self._decay_epsilon()
                return policy_step.PolicyStep(max_logit_index)

            else:
                self._decay_epsilon()
                return self.random_policy.action(time_step)

        else:
            output = self.model.predict(np.expand_dims(time_step.observation, 0))  # ndarray shape (1,4096)
            max_logit_index = np.argmax(output[0])
            return policy_step.PolicyStep(max_logit_index)

    def _decay_epsilon(self):
        if self.epsilon <= self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon *= self.decay
