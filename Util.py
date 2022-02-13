import os
from os.path import exists

import chess
import chess.pgn
import numpy as np
import zarr


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


def process_board(board: chess.Board) -> np.ndarray:
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

    return bitboards_to_array(bitboards)  # shape (8,8,13)


def text_to_action(origin: str, destination: str) -> int:
    # accepts strings of the form 'e2', 'h5', etc
    # separate string for origin and destination of move
    # e.g ('e2', 'e4')
    return chess.SQUARE_NAMES.index(origin) * 64 + chess.SQUARE_NAMES.index(destination)


def move_to_action(move: chess.Move):
    return move.from_square * 64 + move.to_square


def write_data(observations: np.ndarray, moves: np.ndarray,
               create: bool, obs_filepath: str, moves_filepath: str):
    if create:
        z_obs = zarr.array(observations, chunks=(10000, 8, 8, 13))
        z_moves = zarr.array(moves, chunks=10000)
        zarr.save(obs_filepath, z_obs)
        zarr.save(moves_filepath, z_moves)
        print("Z_obs shape: " + str(z_obs.shape))
        print("Z_moves shape: " + str(z_moves.shape))
    else:
        z_obs = zarr.open(obs_filepath, mode="r+")
        z_moves = zarr.open(moves_filepath, mode="r+")
        z_obs.append(observations)
        z_moves.append(moves)
        print("Z_obs shape: " + str(z_obs.shape))
        print("Z_moves shape: " + str(z_moves.shape))


def process_files_to_zarr(MIN_ELO: int, pgn_dir: str, obs_filepath: str, moves_filepath: str, batch_size=30000):
    total_games = 0
    first_file = True

    if exists(obs_filepath) and exists(moves_filepath):
        first_file = False
        total_games += zarr.open(moves_filepath, mode="r").shape[0]

    for filename in os.listdir(pgn_dir):

        observations = []
        moves = []
        file_path = os.path.join(pgn_dir, filename)

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

        total_games += len(games)
        print("Games that meet criteria: " + str(total_games))

        game_count = 0
        for game in games:

            if game_count > batch_size:
                observations = np.array(observations)
                moves = np.array(moves)
                if first_file:
                    write_data(observations, moves, first_file, obs_filepath, moves_filepath)
                    first_file = False
                else:
                    write_data(observations, moves, first_file, obs_filepath, moves_filepath)

                observations = []
                moves = []

            board = game.board()
            for move in game.mainline_moves():
                # shape (_,8,8,13)
                observations.append(process_board(board))
                moves.append(move_to_action(move))
                board.push(move)
            game_count += 1

        pgn.close()
        observations = np.array(observations)
        moves = np.array(moves)

        if first_file:
            write_data(observations, moves, first_file, obs_filepath, moves_filepath)
            first_file = False
        else:
            write_data(observations, moves, first_file, obs_filepath, moves_filepath)


def _split_zarrs(indices: list, z_obs: zarr.array, z_moves: zarr.array, num_to_write: int,
                 obs_filepath: str, moves_filepath: str):

    obs = []
    mov = []
    count = 0
    first = True

    for i in indices:
        obs.append(z_obs[i])
        mov.append(z_moves[i])
        count += 1

        if count > num_to_write:
            count = 0
            obs = np.array(obs)
            mov = np.array(mov)

            if first:
                write_data(obs, mov, first, obs_filepath, moves_filepath)
                first = False
            else:
                write_data(obs, mov, first, obs_filepath, moves_filepath)

            obs = []
            mov = []

    obs = np.array(obs)
    mov = np.array(mov)
    write_data(obs, mov, first, obs_filepath, moves_filepath)


def train_test_split(obs_filepath: str, moves_filepath: str, train_obs_filepath: str,
                     train_moves_filepath: str, test_obs_filepath: str,
                     test_moves_filepath: str, train_frac=0.8, num_to_write=1_000_000):

    z_obs = zarr.open(obs_filepath, mode="r+")
    z_moves = zarr.open(moves_filepath, mode="r+")

    if len(z_obs) != len(z_moves):
        raise ValueError

    indices = np.arange(len(z_moves))
    rng = np.random.default_rng()
    rng.shuffle(indices)

    num_train_examples = int(train_frac * len(z_moves))

    train_indices = indices[0:num_train_examples]
    test_indices = indices[num_train_examples:]

    _split_zarrs(train_indices, z_obs, z_moves, num_to_write, train_obs_filepath, train_moves_filepath)
    _split_zarrs(test_indices, z_obs, z_moves, num_to_write, test_obs_filepath, test_moves_filepath)
