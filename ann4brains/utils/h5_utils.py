import h5py
import numpy as np


def read_h5(file_name, field_names):
    f = h5py.File(file_name, 'r')

    X = {}
    for field_name in field_names:
        file_X = f.get(field_name)
        X[field_name] = (np.copy(file_X))

    f.close()
    return X


def caffe_write_h5(database_filename, X, Y):
    """Writes data to an h5 database in a format expected by caffe.

    Inputs,
        database_filename : (string) the name and path of the file to write to.
        X : (numpy array) rows are samples, columns are dimensions.
        Y : (numpy array) rows are samples, columns are different labels.
    """

    # Make sure the same number of samples are passed in.
    assert X.shape[0] == Y.shape[0]

    # Caffe requires the data and labels to be floats. To save space, might as well have them as float32s.
    assert X.dtype == np.float32
    # assert X_vec.dtype == np.float32
    assert Y.dtype == np.float32

    # Writes data into the hdf5 database.
    # Note that your *.prototxt should match these names.
    with h5py.File(database_filename, 'w') as f:
        f['data'] = X
        # f['vec'] = X_vec
        f['label'] = Y

    # Simply writes a single line that indicates to caffe where the database is stored.
    with open(database_filename + '.txt', 'w') as f:
        f.write(database_filename + '\n')