import os
import random
import sys

import chess
import chess.engine
import ray

import ChessNetwork


class IllegalMoveException(Exception):
    pass


@ray.remote(num_cpus=1)
class ChessWorker:

    def __init__(self):
        self.color = 0
        self.engine = None
        self.board = chess.Board()
        self.setup_engine()

    def setup_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"/home/ricardo/Downloads/stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")
        self.engine.configure({"Hash": 32})
        self.engine.configure({"Threads": 1})

    def reset_board(self):
        self.board.reset_board()

    def play_test(self):
        while not self.board.is_game_over():
            result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            self.board.push(result.move)
            # print(result.move)
        return str(os.getpid()) + ": " + str(self.board.outcome().result())

    def play(self, network: ChessNetwork.ChessNetwork):
        self.color = random.randint(0, 1)

        try:

            if self.color:  # white = 1
                network.build_input(self.board)
                network.predict_move()

            while not self.board.is_game_over():
                result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
                self.board.push(result.move)
                # print(result.move)
                network.build_input(self.board)
                network.predict_move()
            return str(os.getpid()) + ": " + str(self.board.outcome().result())

        except (IllegalMoveException, ValueError):
            print("Exception")
            sys.exit()
