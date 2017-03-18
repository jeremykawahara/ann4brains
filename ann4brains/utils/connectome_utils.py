import numpy as np


def mat2gray(mat):
    """Normalize mat between 0 and 1"""
    mat = (mat - mat.min()) / (mat.max() - mat.min())
    return mat


def normalize_connectomes(X):
    X_norm = []
    for x in X:
        x = mat2gray(x)
        X_norm.append(x)

    assert not np.any(np.isnan(X_norm))
    return np.asarray(X_norm)


def vectorize_symmetric_list(X):
    """Return a [N x D] matrix vectorized of the lower triangular values (without diagonal) of the list X"""

    # Must be smaller than the lowerest value in X.
    sentinal = -1
    assert X.min() > sentinal

    # Assumes all the matrices are the same size.
    mat = X[0]

    # Get the indexes to the upper triangle (since symmetric, we have double the indexes.)
    upper_indexes = np.triu_indices(mat.shape[0])

    # Data matrix where the entries have been vectorized.
    vec_mat = np.zeros((len(X), 4005), dtype=mat.dtype)

    # print vec_mat.shape

    idx = 0
    for mat_X in X:
        # Make sure to copy this otherwise will change the values in place (pass-by-ref).
        mat = np.copy(mat_X)

        # We flag all the uppder indexes including the diagonal.
        mat[upper_indexes] = sentinal

        # Get all the values that are greater than the sentinal (should be only the lower tri without diagonal)
        lower_tri_no_diag = mat[mat > sentinal]
        vec_mat[idx, :] = lower_tri_no_diag
        idx += 1

    return vec_mat
