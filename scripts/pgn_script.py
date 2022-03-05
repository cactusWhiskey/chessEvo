import Util

MIN_ELO = 2000
pgn_directory = '/home/ricardo/Downloads/Pgn'
save_fileZobs = '/home/ricardo/Downloads/pgn_save/obsZ_H.zarr'
save_fileZmoves = '/home/ricardo/Downloads/pgn_save/movesZ_H.zarr'

Util.process_files_to_zarr(MIN_ELO, pgn_directory, save_fileZobs, save_fileZmoves, (10_000, 8, 8, 91), singles=False)
