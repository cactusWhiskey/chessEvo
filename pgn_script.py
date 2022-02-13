import Util

MIN_ELO = 2000
pgn_directory = '/home/ricardo/Downloads/Pgn'
save_file = '/home/ricardo/Downloads/pgn_save/obs.npy'
save_fileZobs = '/home/ricardo/Downloads/pgn_save/obsZ.zarr'
save_fileZmoves = '/home/ricardo/Downloads/pgn_save/movesZ.zarr'

Util.process_files_to_zarr(MIN_ELO, pgn_directory, save_fileZobs, save_fileZmoves)
