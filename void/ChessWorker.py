import os
import random
import chess
import chess.engine
import ray
from deap import creator, base
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import ChessNetwork

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

WHITE = 1
BLACK = 0
MATE_SCORE = 200.0  # just some value to stand in for mate
ILLEGAL_MOVE_PENALTY = -100.0


class IllegalMoveException(Exception):
    pass


@ray.remote(num_cpus=1)
class ChessWorker:

    def __init__(self):
        self.color = 0
        self.engine = None
        self.board = chess.Board()
        self.setup_engine()
        self.network = ChessNetwork.ChessNetwork()

    def setup_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"/home/ricardo/Downloads/stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")
            #r"/home/arjudoso/stockfish_14.1_linux_x64_avx2")
        self.engine.configure({"Hash": 32})
        self.engine.configure({"Threads": 1})

    def reset_board(self):
        self.board.reset()

    def play_test(self):
        scores = []
        moveCount = 0
        while not self.board.is_game_over():
            result = self.engine.play(self.board, chess.engine.Limit(time=0.1), info=chess.engine.INFO_SCORE)
            self.board.push(result.move)
            povScore = result.info['score']
            scores.append(self.processScore(povScore))
            print(result.move)
            # result = self.engine.play(self.board, chess.engine.Limit(time=0.1), info=chess.engine.INFO_SCORE)
            # self.board.push(result.move)
            # print(result.move)
            # moveCount += 1
            # print(str(moveCount))
        return self.eval(moveCount, False, scores), str(os.getpid()) + ": " + str(self.board.outcome().result())

    def play(self, individual):
        self.network.build_from_genome(individual)
        self.color = random.randint(0, 1)
        moveCount = 0
        scores = [0.0]
        try:

            if self.color:  # white = 1, black = 0
                self.network.build_input(self.board)
                move = self.network.predict_move()
                if not self.board.is_legal(move):
                    raise IllegalMoveException
                self.board.push(move)
                moveCount += 1

            while not self.board.is_game_over():
                result = self.engine.play(self.board, chess.engine.Limit(time=0.1), info=chess.engine.INFO_SCORE)
                povScore = result.info['score']
                scores.append(self.processScore(povScore))
                self.board.push(result.move)

                self.network.build_input(self.board)
                move = self.network.predict_move()
                if not self.board.is_legal(move):
                    raise IllegalMoveException
                self.board.push(move)
                moveCount += 1  # keep track of number of legal moves network makes

            return self.eval(moveCount, False, scores),

        except (IllegalMoveException, ValueError):
            return self.eval(moveCount, True, scores),

    def eval(self, moveCount: int, illegal: bool, scores: list):
        # print("Moves: " + str(moveCount) + " Avg CP: " + str(np.average(scores)))
        if illegal:
            # print("Illegal")
            return moveCount #+ np.average(scores) + ILLEGAL_MOVE_PENALTY
        # print("Moves: " + str(moveCount) + " Avg CP: " + str(np.average(scores)))
        return moveCount #+ np.average(scores)

    def processScore(self, povScore: chess.engine.PovScore):
        # povScore is always from the point of view of the opponent, so positive score is opponent
        # winning, negative score is us winning
        score = 0.0
        if povScore.is_mate():  # some is being mated
            if povScore.relative.moves > 0:  # oh, we are getting mated
                score = -1 * MATE_SCORE
            else:  # we are doing the mating
                score = MATE_SCORE
        else:  # position isn't a forced mate
            score = (povScore.relative.cp * -1.0) / 100.0

        return score

    def close_engine(self):
        if self.engine is not None:
            self.engine.quit()

