from enum import Enum
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import chess
import chess.engine
import random
import Util

MATE_SCORE = 200.0  # just some value to stand in for mate
ILLEGAL_MOVE_PENALTY = -100.0
WIN_BONUS = 50.0


class Color(Enum):
    WHITE = 0
    BLACK = 1


promotion_moves = ["a7a8", "a7b8",
                   "b7a8", "b7b8", "b7c8",
                   "c7b8", "c7c8", "c7d8",
                   "d7c8", "d7d8", "d7e8",
                   "e7d8", "e7e8", "e7f8",
                   "f7e8", "f7f8", "f7g8",
                   "g7f8", "g7g8", "g7h8",
                   "h7g8", "h7h8",

                   "a2a1", "a2b1",
                   "b2a1", "b2b1", "b2c1",
                   "c2b1", "c2c1", "c2d1",
                   "d2c1", "d2d1", "d2e1",
                   "e2d1", "e2e1", "e2f1",
                   "f2e1", "f2f1", "f2g1",
                   "g2f1", "g2g1", "g2h1",
                   "h2g1", "h2h1"]

promotion_types = ["n", "b", "r", "q"]


class ChessEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=4271, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8, 8, 13), dtype=np.uint8, minimum=0, maximum=1, name='observation')

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
        self.print = False
        self.total_reward = 0

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

        if self.print is True:
            print(self._board)
            print(povScore)
            print("Engine is: " + str(self._color))

    def close(self) -> None:
        self.close_engine()

    def close_engine(self):
        if self.engine is not None:
            self.engine.quit()

    def engine_self_play(self):
        while not self._board.is_game_over():
            result = self.engine.play(self._board, chess.engine.Limit(time=0.1))
            self._board.push(result.move)
            print(result.move)
        print(self._board.result())

    def self_test(self, num_runs=1, random_color=False):
        if random_color is True:
            color = random.randint(0, 1)
            if color != self._color:
                self.change_color()

        for i in range(num_runs):
            self.reset()
            print_counter = 0
            while not self._board.is_game_over():
                result = self.engine.play(self._board, chess.engine.Limit(time=0.1))
                print(result.move)
                self.step(self.move_to_action(result.move))
                print_counter += 1
                if print_counter % 5 == 0:
                    print(self._board)
            print(self._board.outcome().result())

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
        self.total_reward = 0
        return ts.restart(self._observation)

    def _step(self, action: int):

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
            # print("legal")
            self.total_reward += 1
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
                    print(self.outcome)

        # reward = self._calculate_reward()
        reward = self.total_reward
        self._build_observation()
        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward)

    def _build_observation(self):
        self._observation = []
        self._observation = Util.process_board(self._board)  # shape (8,8,13)

    def _build_move(self, action: int):

        if action > 4095:  # everything > 4095 is a promotion move
            moveStr = self._get_promotion_move(action)
        else:
            origin_square = int(action / 64)
            destination_square = action % 64
            if origin_square == destination_square:
                moveStr = "0000"  # null move
            else:
                moveStr = chess.SQUARE_NAMES[origin_square] + chess.SQUARE_NAMES[destination_square]
        move = chess.Move.from_uci(moveStr)
        self._move = move
        self._legal = self._board.is_legal(move)

        # print("Env internals: ")
        # print(self._legal)
        # print(self._board)
        # print(move)
        # print(action)

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

    @staticmethod
    def _get_promotion_move(action: int):
        # action between 4096 and 4271 inclusive
        diff = action - 4096  # scales to 0 - 175
        move_index = diff // 4
        type_index = diff % 4

        move = promotion_moves[move_index]
        promotion = promotion_types[type_index]
        return move + promotion
