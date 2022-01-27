import chess
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(8, 8)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4096)
    ])
    return model


class ChessNetwork:
    def __init__(self):
        self.move = None
        self.model = build_model()
        self.input = []
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def predict_move(self):
        output = self.probability_model.predict(np.expand_dims(self.input, 0))  # ndarray shape (1,4096)
        max_logit = np.argmax(output[0])
        origin_square = int(max_logit / 64)
        destination_square = max_logit % 64
        moveStr = chess.SQUARE_NAMES[origin_square] + chess.SQUARE_NAMES[destination_square]
        self.move = chess.Move.from_uci(moveStr)
        return self.move

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
        self.input = np.array(self.input).reshape(8, 8)

    def build_from_genome(self, individual: list):
        i = 0
        for layer in self.model.layers[1:]:
            layer.set_weights(individual[i])
            i += 1
