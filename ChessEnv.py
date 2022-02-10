from enum import Enum
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import chess
import chess.engine
import random

MATE_SCORE = 200.0  # just some value to stand in for mate
ILLEGAL_MOVE_PENALTY = -100.0
WIN_BONUS = 50.0


# Credit Mateen Ulhaq
#  https://chess.stackexchange.com/questions/29294/quickly-converting-board-to-bitboard-representation-using-python-chess-library
def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    # returns an array of (8,8) arrays. The (8,8) arrays have index [0][0] = Square A1,
    # index [7][7] = H8
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(0, 8, 1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    b = b.reshape(-1, 8, 8)
    return np.transpose(b, (1, 2, 0))


class Color(Enum):
    WHITE = 0
    BLACK = 1


class ChessEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4095, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8, 8, 13), dtype=np.float64, minimum=-6.0, maximum=6.0, name='observation')

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
            print("legal")
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

        # reward = self._calculate_reward()
        reward = self.total_reward
        self._build_observation()
        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward)

    def _build_observation(self):
        self._observation = []

        # Black side is most significant bits
        bitboards = [
            int(self._board.pieces(chess.PAWN, chess.BLACK)),
            int(self._board.pieces(chess.KNIGHT, chess.BLACK)),
            int(self._board.pieces(chess.BISHOP, chess.BLACK)),
            int(self._board.pieces(chess.ROOK, chess.BLACK)),
            int(self._board.pieces(chess.QUEEN, chess.BLACK)),
            int(self._board.pieces(chess.KING, chess.BLACK)),

            int(self._board.pieces(chess.PAWN, chess.WHITE)),
            int(self._board.pieces(chess.KNIGHT, chess.WHITE)),
            int(self._board.pieces(chess.BISHOP, chess.WHITE)),
            int(self._board.pieces(chess.ROOK, chess.WHITE)),
            int(self._board.pieces(chess.QUEEN, chess.WHITE)),
            int(self._board.pieces(chess.KING, chess.WHITE)),

            self._board.castling_rights
        ]

        if self._board.turn:  # White to move
            bitboards[-1] = bitboards[
                                -1] | 4  # flip a bit on the white side of the castling rights board to indicate turn order

        if self._board.ep_square is not None:
            ep_bb = 2 ** self._board.ep_square
            bitboards[-1] = bitboards[-1] | ep_bb  # add the ep square to the castling rights bb

        bitboards = np.array(bitboards, dtype=np.uint64)

        self._observation = bitboards_to_array(bitboards)  # shape (8,8,13)

    def _build_move(self, action):
        origin_square = int(action / 64)
        destination_square = action % 64
        if origin_square == destination_square:
            moveStr = "0000"  # null move
        else:
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

    @staticmethod
    def text_to_action(origin: str, destination: str) -> int:
        # accepts strings of the form 'e2', 'h5', etc
        # separate string for origin and destination of move
        # e.g ('e2', 'e4')
        return chess.SQUARE_NAMES.index(origin) * 64 + chess.SQUARE_NAMES.index(destination)

    @staticmethod
    def move_to_action(move: chess.Move):
        return move.from_square * 64 + move.to_square
