import Util

save_fileZobs = '/home/ricardo/Downloads/pgn_save/obsZ.zarr'
save_fileZmoves = '/home/ricardo/Downloads/pgn_save/movesZ.zarr'

save_fileZobs_train = '/home/ricardo/Downloads/pgn_save/obsZ_train.zarr'
save_fileZmoves_train = '/home/ricardo/Downloads/pgn_save/movesZ_train.zarr'
save_fileZobs_test = '/home/ricardo/Downloads/pgn_save/obsZ_test.zarr'
save_fileZmoves_test = '/home/ricardo/Downloads/pgn_save/movesZ_test.zarr'

Util.train_test_split(save_fileZobs, save_fileZmoves, save_fileZobs_train, save_fileZmoves_train, save_fileZobs_test,
                      save_fileZmoves_test, num_to_write=100_000)
