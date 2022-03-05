import os
from collections import deque
from os.path import exists
import chess
import chess.pgn
import numpy as np
import zarr
from ML import ChessEnv

history = deque(maxlen=7)


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


def process_board(board: chess.Board, new_game: bool = False, single=True) -> np.ndarray:
    if new_game:
        history.clear()
        for i in range(7):
            history.append(np.zeros((8, 8, 13), dtype=np.uint8))

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
    bitboards = bitboards_to_array(bitboards)  # shape (8,8,13)

    if single:
        return bitboards  # shape (8,8,13)

    history.append(bitboards)
    return np.concatenate(history, axis=2)  # shape (8,8,91)


def text_to_action(origin: str, destination: str) -> int:
    # accepts strings of the form 'e2', 'h5', etc
    # separate string for origin and destination of move
    # e.g ('e2', 'e4')
    return chess.SQUARE_NAMES.index(origin) * 64 + chess.SQUARE_NAMES.index(destination)


def move_to_action(move: chess.Move):
    if move.promotion is None:
        return move.from_square * 64 + move.to_square
    else:
        promotion_index = move.promotion - 2  # 0 to 3
        move_index = ChessEnv.promotion_moves.index(move.__str__()[0:4])  # 0 to 43

        action = move_index * 4 + promotion_index  # 0 to 175
        action += 4096  # now range is 4096 to 4271 inclusive

        return action


def write_array(data: np.ndarray, create: bool,
                filepath: str, chunks=None, verbose=True):
    if create:
        z_data = zarr.array(data, chunks=chunks)
        zarr.save(filepath, z_data)
    else:
        z_data = zarr.open(filepath, mode="r+")
        z_data.append(data)

    if verbose:
        # print("Saved to: " + filepath)
        print("zarr_data shape: " + str(z_data.shape))


# def write_data(observations: np.ndarray, moves: np.ndarray,
#                create: bool, obs_filepath: str, moves_filepath: str):
#     if create:
#         z_obs = zarr.array(observations, chunks=(10000, 8, 8, 13))
#         z_moves = zarr.array(moves, chunks=10000)
#         zarr.save(obs_filepath, z_obs)
#         zarr.save(moves_filepath, z_moves)
#         print("Z_obs shape: " + str(z_obs.shape))
#         print("Z_moves shape: " + str(z_moves.shape))
#     else:
#         z_obs = zarr.open(obs_filepath, mode="r+")
#         z_moves = zarr.open(moves_filepath, mode="r+")
#         z_obs.append(observations)
#         z_moves.append(moves)
#         print("Z_obs shape: " + str(z_obs.shape))
#         print("Z_moves shape: " + str(z_moves.shape))


def process_games(games: list, first_file: bool, inputs_filepath, labels_filepath,
                  obs_chunks, singles=True):

    observations = []
    moves = []

    for game in games:
        board = game.board()
        first_move = True

        for move in game.mainline_moves():
            if move.uci() == "0000":
                break
            observations.append(process_board(board, first_move, single=singles))
            moves.append(move_to_action(move))
            board.push(move)
            first_move = False

    observations = np.array(observations)
    moves = np.array(moves)

    if first_file:
        write_array(observations, first_file, inputs_filepath, chunks=obs_chunks)
        write_array(moves, first_file, labels_filepath, chunks=10000)
        first_file = False
    else:
        write_array(observations, first_file, inputs_filepath, chunks=obs_chunks)
        write_array(moves, first_file, labels_filepath, chunks=10000)

    return first_file


def process_files_to_zarr(MIN_ELO: int, pgn_dir: str, obs_filepath: str, moves_filepath: str,
                          obs_chunks, batch_size=2_000, singles=True):

    first = True

    if exists(obs_filepath) and exists(moves_filepath):
        first = False
        # total_games += zarr.open(moves_filepath, mode="r").shape[0]

    for filename in os.listdir(pgn_dir):
        total_games = 0

        file_path = os.path.join(pgn_dir, filename)
        print("Processing file: " + filename)
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

            if len(games) >= batch_size:
                print("Batch size reached")
                total_games += len(games)
                first = process_games(games, first, obs_filepath, moves_filepath, obs_chunks, singles=singles)
                games = []

        total_games += len(games)
        first = process_games(games, first, obs_filepath, moves_filepath, obs_chunks, singles=singles)

        print("Games that meet criteria: " + str(total_games))
