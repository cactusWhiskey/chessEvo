import zarr

import Util
import zarr_gen

save_fileZobs = '/home/ricardo/Downloads/pgn_save/obsZ_H.zarr'
save_fileZmoves = '/home/ricardo/Downloads/pgn_save/movesZ_H.zarr'

x_train_file = '/home/ricardo/Downloads/pgn_save/obsZ_train_H.zarr'
y_train_file = '/home/ricardo/Downloads/pgn_save/movesZ_train_H.zarr'
x_test_file = '/home/ricardo/Downloads/pgn_save/obsZ_test_H.zarr'
y_test_file = '/home/ricardo/Downloads/pgn_save/movesZ_test_H.zarr'

first = True

z_inputs = zarr.open(save_fileZobs, mode="r")
z_labels = zarr.open(save_fileZmoves, mode="r")

gen = zarr_gen.ZarrGen(z_inputs, z_labels, split=True, num_to_load=100_000)

for batch in gen:
    x_train, y_train, x_test, y_test = batch

    if first:
        print("X Train: ")
        Util.write_array(x_train, True, x_train_file, chunks=(10000, 8, 8, 91))
        print("Y Train: ")
        Util.write_array(y_train, True, y_train_file, chunks=10000)
        print("X Test: ")
        Util.write_array(x_test, True, x_test_file, chunks=(10000, 8, 8, 91))
        print("Y Test: ")
        Util.write_array(y_test, True, y_test_file, chunks=10000)
        first = False
    else:
        print("X Train: ")
        Util.write_array(x_train, False, x_train_file, chunks=(10000, 8, 8, 91))
        print("Y Train: ")
        Util.write_array(y_train, False, y_train_file, chunks=10000)
        print("X Test: ")
        Util.write_array(x_test, False, x_test_file, chunks=(10000, 8, 8, 91))
        print("Y Test: ")
        Util.write_array(y_test, False, y_test_file, chunks=10000)

