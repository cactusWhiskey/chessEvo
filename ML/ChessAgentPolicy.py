import numpy as np

import Util
from ML.ChessEnv import ChessEnv
from ML.DqnAgentPolicy import DqnAgentPolicy
from keras import Model
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class ChessAgentPolicy(DqnAgentPolicy):
    def __init__(self, time_step_spec: ts.TimeStep, action_spec: types.NestedArraySpec,
                 model: Model, env: ChessEnv, force_legal=True):
        super().__init__(time_step_spec, action_spec, model, epsilon_greedy=False)

        self.force_legal = force_legal
        self.env = env

    def _get_max_logit(self, time_step: ts.TimeStep):
        output = self.model.predict(np.expand_dims(time_step.observation, 0))  # ndarray shape (1,4272)

        if self.force_legal:
            #print("Policy internals: ")
            #print(self.env._board)
            board = self.env._board
            legal_moves = board.legal_moves

            if len(list(legal_moves)) == 0:
                return 0

            legal_actions = []
            for move in legal_moves:
                legal_actions.append(Util.move_to_action(move))

            legal_outputs = (output[0])[legal_actions]
            max_logit_index = np.argmax(legal_outputs)
            action = legal_actions[max_logit_index]
            #print("Action: " + str(action))
            return action

        else:
            return np.argmax(output[0])
