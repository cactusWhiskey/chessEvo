import os
import chess
import chess.pgn
import numpy as np
import zarr as zarr

import ChessEnv

MIN_ELO = 2000
pgn_directory = '/home/ricardo/Downloads/Pgn'
save_file = '/home/ricardo/Downloads/pgn_save/obs.npy'
save_fileZ = '/home/ricardo/Downloads/pgn_save/obsZ.zarr'


def process_board(board: chess.Board):
    # Black side is most significant bits
    bitboards = [
        int(board.pieces(chess.PAWN, chess.BLACK)),
        int(board.pieces(chess.KNIGHT, chess.BLACK)),
        int(board.pieces(chess.BISHOP, chess.BLACK)),
        int(board.pieces(chess.ROOK, chess.BLACK)),
        int(board.pieces(chess.QUEEN, chess.BLACK)),
        int(board.pieces(chess.KING, chess.BLACK)),

        int(board.pieces(chess.PAWN, chess.WHITE)),
        int(board.pieces(chess.KNIGHT, chess.WHITE)),
        int(board.pieces(chess.BISHOP, chess.WHITE)),
        int(board.pieces(chess.ROOK, chess.WHITE)),
        int(board.pieces(chess.QUEEN, chess.WHITE)),
        int(board.pieces(chess.KING, chess.WHITE)),

        board.castling_rights
    ]

    if board.turn:  # White to move
        bitboards[-1] = bitboards[
                            -1] | 4  # flip a bit on the white side of the castling rights board to indicate turn order

    if board.ep_square is not None:
        ep_bb = 2 ** board.ep_square
        bitboards[-1] = bitboards[-1] | ep_bb  # add the ep square to the castling rights bb

    bitboards = np.array(bitboards, dtype=np.uint64)

    return ChessEnv.bitboards_to_array(bitboards)  # shape (8,8,13)


observations = []

for filename in os.listdir(pgn_directory):
    file_path = os.path.join(pgn_directory, filename)

    pgn = open(file_path)

    offsets = []

    while True:
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)

        if headers is None:
            break

        elos = [headers["WhiteElo"], headers["BlackElo"]]
        for index, elo in enumerate(elos):
            if elo == "?":
                elos[index] = 0
        min_elo = min(int(elos[0]), int(elos[1]))

        if min_elo > MIN_ELO:
            offsets.append(offset)

    games = []
    for offset in offsets:
        pgn.seek(offset)
        games.append(chess.pgn.read_game(pgn))

    print(str(len(games)))

    for game in games:
        board = game.board()
        observations.append(process_board(board))  # shape (_,8,8,13)

        for move in game.mainline_moves():
            board.push(move)
            observations.append(process_board(board))

    pgn.close()

observations = np.array(observations)
print(observations.shape)
np.save(save_file, observations)
z1 = zarr.save(save_fileZ, observations)
