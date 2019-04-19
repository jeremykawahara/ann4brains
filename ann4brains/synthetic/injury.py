from __future__ import print_function
import os
import scipy.io
import numpy as np


class ConnectomeInjury(object):

    def __init__(self,
                 base_filename=os.path.join('data', 'base.mat'),  # Base connectome filename.
                 n_injuries=2,  # How many injuries to have (currently only works for 2).
                 signature_seed=333,  # Random seed for the injury signature.
                 ):
        """Use to create synthetic injury data."""

        # Set the mean base connectome.
        self.X_mn = self.load_base_connectome(base_filename)

        # Generate the injury signatures (set the random seed so get same signatures).
        r_state = np.random.RandomState(signature_seed)
        self.sigs = self.generate_injury_signatures(self.X_mn, n_injuries, r_state)

    def generate_injury(self, n_samples=100,  # How many samples to create.
                        noise_weight=0.125,  # How much to weigh the noise.
                        ):
        """Return n_samples of synthetic injury data and corresponding injury strength."""

        # Generate phantoms with injuries of different strengths (and add noise)
        # TODO: allow for N injury patterns
        X, Y = self.sample_injury_strengths(n_samples, self.X_mn, self.sigs[0],
                                            self.sigs[1], noise_weight)

        # Make sure the number of samples matches what was specified.
        assert X.shape[0] == n_samples
        assert Y.shape[0] == n_samples

        return X, Y


    @staticmethod
    def load_base_connectome(file_name, verbose=False):
        """Loads the connectome that serves as the base of the synthetic data"""

        # Load the data.
        X_mn = scipy.io.loadmat(file_name)
        X_mn = X_mn['X_mn']

        if verbose:
            print('Data shape: ', X_mn.shape, ' Min value: ', X_mn.min(), ' Max value: ', X_mn.max())
        return X_mn

    @staticmethod
    def generate_injury_signatures(X_mn, n_injuries, r_state):
        """Generates the signatures that represent the underlying signal in our synthetic experiments.

        d : (integer) the size of the input matrix (assumes is size dxd)
        """

        # Get the strongest regions, which we will apply simulated injuries
        sig_indexes = get_k_strongest_regions(X_mn, n_injuries, verbose=False)
        d = X_mn.shape[0]

        S = []

        # Create a signature for
        for idx, sig_idx in enumerate(sig_indexes):
            # Okay, let's make some signature noise vectors.
            A_vec = r_state.rand((d))
            # B_vec = np.random.random((n))

            # Create the signature matrix.
            A = np.zeros((d, d))
            A[:, sig_idx] = A_vec
            A[sig_idx, :] = A_vec
            S.append(A)

            assert (A.T == A).all()  # Check if matrix is symmetric.

        return np.asarray(S)

    @staticmethod
    def sample_injury_strengths(n_samples, X_mn, A, B, noise_weight):
        """Returns n_samples connectomes with simulated injury from two sources."""
        mult_factor = 10

        n_classes = 2

        # Range of values to predict.
        n_start = 0.5
        n_end = 1.4
        # amt_increase = 0.1

        # These will be our Y.
        A_weights = np.random.uniform(n_start, n_end, [n_samples])
        B_weights = np.random.uniform(n_start, n_end, [n_samples])

        X_h5 = np.zeros((n_samples, 1, X_mn.shape[0], X_mn.shape[1]), dtype=np.float32)
        Y_h5 = np.zeros((n_samples, n_classes), dtype=np.float32)

        for idx in range(n_samples):
            w_A = A_weights[idx]
            w_B = B_weights[idx]

            # Get the matrix.
            X_sig = apply_injury_and_noise(X_mn, A, w_A * mult_factor, B, w_B * mult_factor, noise_weight)

            # Normalize.
            X_sig = (X_sig - X_sig.min()) / (X_sig.max() - X_sig.min())

            # Put in h5 format.
            X_h5[idx, 0, :, :] = X_sig
            Y_h5[idx, :] = [w_A, w_B]

        return X_h5, Y_h5


def get_symmetric_noise(m, n):
    """Return a random noise image of size m x n with values between 0 and 1."""

    # Generate random noise image.
    noise_img = np.random.rand(m, n)

    # Make the noise image symmetric.
    noise_img = noise_img + noise_img.T

    # Normalize between 0 and 1.
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    assert noise_img.max() == 1  # Make sure is between 0 and 1.
    assert noise_img.min() == 0
    assert (noise_img.T == noise_img).all()  # Make sure symmetric.

    return noise_img


def simulate_injury(X, weight_A, sig_A, weight_B, sig_B):
    denom = (np.ones(X.shape) + (weight_A * sig_A)) * (np.ones(X.shape) + (weight_B * sig_B))
    X_sig_AB = np.divide(X, denom)
    return X_sig_AB


def apply_injury_and_noise(X, Sig_A, weight_A, Sig_B, weight_B, noise_weight):
    """Returns a symmetric, signed, noisy, adjacency matrix with simulated injury from two sources."""

    X_sig_AB = simulate_injury(X, weight_A, Sig_A, weight_B, Sig_B)

    # Get the noise image.
    noise_img = get_symmetric_noise(X.shape[0], X.shape[1])

    # Weight the noise image.
    weighted_noise_img = noise_img * noise_weight

    # Add the noise to the original image.
    X_sig_AB_noise = X_sig_AB + weighted_noise_img

    assert (X_sig_AB_noise.T == X_sig_AB_noise).all()  # Make sure still is symmetric.

    return X_sig_AB_noise


def get_k_strongest_regions(X, k, verbose=False):
    """Return the k regions (matrix columns) with the highest median values."""

    # Make a copy of this array, since we will modify it, and will change the orginal X.
    X = np.copy(X)

    highest_col_indexes = []

    # We combine the column based on the median value.
    for idx in range(k):
        # sum_cols = np.sum(X_mn, axis=0)
        sum_cols = np.median(X, axis=0)  #
        max_idx = np.argmax(sum_cols)
        highest_col_indexes.append(max_idx)

        # Zero out the largest column so we can find the next largest one.
        X[:, max_idx] = 0
        if verbose:
            print("%i => column index of largest averaged value: %i" % (idx, max_idx))

    return highest_col_indexes

"""
def generate_injury_data(test_name, X_mn, S, n_samples=100, noise_weight=0.125,
                         dir_syn_data='./generated_synthetic_data'):
    #Generate and write to disk synthetic injury data. Return the file path.

    # Generate phantoms with injuries of different strengths (and add noise)
    X, Y = sample_injury_strengths(n_samples, X_mn, S[0], S[1], noise_weight)  # TODO: allow for N injury patterns

    # Make sure the number of samples matches what was specified.
    assert X.shape[0] == n_samples
    assert Y.shape[0] == n_samples

    # Vectorize connectome.
    X_vec = vectorize_symmetric_list(np.squeeze(X))

    # Write to disk in h5 format for caffe.
    f_name = test_name + '-' + str(n_samples) + '_noise-' + str(noise_weight) + '.h5'
    file_name = os.path.abspath(os.path.join(dir_syn_data, f_name))
    caffe_write_h5(file_name, X, X_vec, Y)
    return file_name
"""

