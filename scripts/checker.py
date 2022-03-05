import zarr
import numpy as np
import Util
import zarr_gen

save_fileZobs = '/home/ricardo/Downloads/pgn_save/obsZ.zarr'
save_fileZmoves = '/home/ricardo/Downloads/pgn_save/movesZ.zarr'

x_train_file = '/home/ricardo/Downloads/pgn_save/obsZ_train.zarr'
y_train_file = '/home/ricardo/Downloads/pgn_save/movesZ_train.zarr'
x_test_file = '/home/ricardo/Downloads/pgn_save/obsZ_test.zarr'
y_test_file = '/home/ricardo/Downloads/pgn_save/movesZ_test.zarr'

z_inputs = zarr.open(save_fileZobs, mode="r")
z_labels = zarr.open(save_fileZmoves, mode="r")

length = len(z_labels)
counter = 0
num_to_load = 500_000

while counter < length:
    #print("Processed: " + str(counter))
    n_labels = z_labels[counter: counter+num_to_load]
    counter += num_to_load

    for i in range(len(n_labels)):
        label = n_labels[i]

        if (label > 4271) or (label < 0):
            print("Label out of range: " + str(label))
            print(str(i))
        # if label > 4095:
        #     print(str(label))
        #     print(str(i))
        if not isinstance(label, np.int64):
            print("Not int: " + str(label))
            print(str(i))
        if (label is None):
            print(str(label))
            print(str(i))
