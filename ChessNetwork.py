import chess
import tensorflow as tf
import numpy as np


class ChessNetwork:
    def __init__(self):
        self.move = None
        self.model = None
        self.input = []

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(8, 8)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])

    def predict_move(self):
        output = self.model.predict(self.input)  # ndarray shape (1,2)
        output = np.round(output * 63.0)

        moveStr = chess.SQUARE_NAMES[int(output[0, 0])] + chess.SQUARE_NAMES[int(output[0, 1])]
        self.move = chess.Move.from_uci(moveStr)

    def build_input(self, board: chess.Board):
        self.input = []
        for i in range(64):
            piece = board.piece_at(i)
            if piece is None:
                p = 0.0
            else:
                p = piece.piece_type / 6.0
                if not piece.color:
                    p = p * -1
            self.input.append(p)
