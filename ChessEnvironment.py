import numpy as np
from enum import Enum
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
import chess
import chess.engine
from tf_agents.trajectories import time_step as ts

MATE_SCORE = 200.0  # just some value to stand in for mate
ILLEGAL_MOVE_PENALTY = -100.0
WIN_BONUS = 50.0


class Color(Enum):
    WHITE = 0
    BLACK = 1


class ChessEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4095, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8, 8), dtype=np.float64, minimum=-6.0, maximum=6.0, name='observation')

        self._board = chess.Board()
        self._color = Color.BLACK
        self._episode_ended = False
        self._observation = None
        self._build_observation()
        self._move = None
        self._legal = None
        self.engine = None
        self._setup_engine()
        self._scores = []
        self.outcome = None
        self._move_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def change_color(self):
        # will flip sides and then make a move, so that the environment is always
        # in a state of waiting for the agent to move
        if self._color is Color.WHITE:
            self._color = Color.BLACK
        else:
            self._color = Color.WHITE

        self.engine_move()
        self._build_observation()

    def engine_move(self, move_time=0.1):
        result = self.engine.play(self._board, chess.engine.Limit(time=move_time),
                                  info=chess.engine.INFO_SCORE)
        povScore = result.info['score']
        self._scores.append(self._process_score(povScore))
        self._board.push(result.move)

    def close_engine(self):
        if self.engine is not None:
            self.engine.quit()

    def engine_self_play(self):
        while not self._board.is_game_over():
            result = self.engine.play(self._board, chess.engine.Limit(time=0.1))
            self._board.push(result.move)
            print(result.move)
        print(self._board.result())

    def _reset(self):
        # does not change engine color
        self._board.reset()
        self._episode_ended = False
        self._build_observation()
        self._scores = []
        self.outcome = None
        self._move_count = 0
        self._move = None
        self._legal = None
        return ts.restart(self._observation)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # build move from action
        self._build_move(action)

        # check legality first
        if not self._legal:
            self._episode_ended = True
            self._scores.append(0.0)
        else:  # is legal move
            self._board.push(self._move)
            self._move_count += 1
            if self._board.is_game_over():
                self._episode_ended = True
                self.outcome = self._board.outcome()
                self._scores.append(self._scores[-1])
            else:  # game not over, have engine make a move
                self.engine_move()
                if self._board.is_game_over():
                    self._episode_ended = True
                    self.outcome = self._board.outcome()

        reward = self._calculate_reward()
        self._build_observation()
        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward)

    def _build_observation(self):
        self._observation = []
        for i in range(64):
            piece = self._board.piece_at(i)
            if piece is None:
                p = 0.0
            else:
                p = float(piece.piece_type)
                if not piece.color:
                    p = p * -1.0
            self._observation.append(p)
        self._observation = np.array(self._observation).reshape(8, 8)

    def _build_move(self, action):
        origin_square = int(action / 64)
        destination_square = action % 64
        moveStr = chess.SQUARE_NAMES[origin_square] + chess.SQUARE_NAMES[destination_square]
        move = chess.Move.from_uci(moveStr)
        self._move = move
        self._legal = self._board.is_legal(move)

    def _setup_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"/home/ricardo/Downloads/stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")
        # r"/home/arjudoso/stockfish_14.1_linux_x64_avx2")
        self.engine.configure({"Hash": 32})
        self.engine.configure({"Threads": 1})

    def _calculate_reward(self):
        if not self._legal:
            return self._move_count + self._scores[-1] + ILLEGAL_MOVE_PENALTY
        if self._episode_ended and self._color == self.outcome.winner:  # agent won!
            return self._move_count + self._scores[-1] + WIN_BONUS
        return self._move_count + self._scores[-1]  # the stockfish won

    @staticmethod
    def _process_score(povScore: chess.engine.PovScore):
        # povScore is always from the point of view of the engine, so positive score is engine
        # winning, negative score is agent winning
        score = 0.0
        if povScore.is_mate():  # someone is being mated
            if povScore.relative.moves > 0:  # agent is getting mated
                score = -1 * MATE_SCORE
            else:  # agent is doing the mating
                score = MATE_SCORE
        else:  # position isn't a forced mate
            score = (povScore.relative.cp * -1.0) / 100.0

        return score


def text_to_action(origin: str, destination: str) -> int:
    # accepts strings of the form 'e2', 'h5', etc
    # separate string for origin and destination of move
    # e.g ('e2', 'e4')
    return chess.SQUARE_NAMES.index(origin) * 64 + chess.SQUARE_NAMES.index(destination)
