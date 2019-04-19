import numpy as np


def init_weights(k, p):
    """Return a random weighting for k instances raised to the p power.

    Args:
        k: A scalar value indicating how many weights to return.
        p: A scalar to raise the weights to.
        A value >1 encourages higher weights for a few and lower weights for the others.

    Returns:
        w: a k-vector of weights that sum to 1.
    """
    w = np.random.rand(k)
    w = w ** p
    w = w / sum(w)

    return w


def _test_init_weights():
    w = init_weights(5, 1)

    # Weights sum to 1.
    assert np.allclose(sum(w), 1)
    # Weights are all positive.
    assert np.alltrue(w > 0)

    # A single weight should get a value of 1.
    w = init_weights(1, 5)
    assert w == 1


def label_distance(target_y, all_labels):
    """Euclidean distance between the a label y and other labels Y.

    Args:
        target_y: A vector representing the labels for a single sample.
        all_labels: A matrix of samples to compare to, where the ith sample is Y[i,:]

    Returns:
        The distance between the target label and all other labels (including the target).
    """

    # The dimensions should be the same.
    assert target_y.shape[0] == all_labels.shape[1]

    return (np.sum(target_y - all_labels, axis=1, keepdims=True) ** 2) ** 0.5


def test_label_distance():
    # Distance between the equal labels should be 0.
    assert label_distance(np.asarray([1]), np.asarray([[1]])) == 0
    assert label_distance(np.asarray([1, 1]), np.asarray([[1, 1]])) == 0

    # Labels with unequal elements should throw an error.
    # print label_distance(np.asarray([1,1]),np.asarray([1]))

    # A more incorrect label should give a higher error.
    assert label_distance(np.asarray([1, 1]), np.asarray([[0, 0]])) > label_distance(np.asarray([1, 1]),
                                                                                     np.asarray([[1, 0]]))
    assert label_distance(np.asarray([1, 1]), np.asarray([[-1, -1]])) > label_distance(np.asarray([1, 1]),
                                                                                       np.asarray([[2, 2]]))

    # Example of how compare multi-labels.
    d = label_distance(np.asarray([0, 0]), np.asarray([[1, 1], [0, 0], [-1, -1]]))
    assert d.shape[0] == 3  # We compared to 3 labels, so the returned rows should be 3.
    assert d[0] == d[2]  # First and last samples should be the same distance apart.


def combine_neighbours(W, X_neighs, Y_neighs):
    """Returns a combined X and Y weighted by the neighbours and W."""

    x_aug = np.zeros((X_neighs.shape[1:]))
    y_aug = np.zeros((Y_neighs.shape[1]))

    for w, x, y in zip(W, X_neighs, Y_neighs):
        x_aug = x_aug + (x * w)
        y_aug = y_aug + (y * w)

    return x_aug, y_aug


def _test_combine_neighbours():
    W = [0.1, 0.4, 0.5]
    X_neighs = X0[0:3, :]
    Y_neighs = Y0[0:3, :]
    x_aug, y_aug = combine_neighbours(W, X_neighs, Y_neighs)
    # plt.subplot(1,2,1); plt.imshow(X_neighs[0,:], interpolation='none'); plt.colorbar(); plt.title(Y_neighs[0,:])
    # plt.subplot(1,2,2); plt.imshow(x_aug,interpolation='none'); plt.colorbar(); plt.title(y_aug)

    # Since we are weighting the neighbours, the max and min should never get larger.
    assert x_aug.max() <= X_neighs.max()
    assert y_aug.max() <= Y_neighs.max()
    assert y_aug.min() >= Y_neighs.min()

    # Try just a single neighbour.
    W = [1.]
    X_neighs = X0[:1, :]
    Y_neighs = Y0[:1, :]
    x_aug, y_aug = combine_neighbours(W, X_neighs, Y_neighs)
    # plt.figure()
    # plt.subplot(1,2,1); plt.imshow(X_neighs[0,:], interpolation='none'); plt.colorbar(); plt.title(Y_neighs[0,:])
    # plt.subplot(1,2,2); plt.imshow(x_aug,interpolation='none'); plt.colorbar(); plt.title(y_aug)

    # The augmented should be the same as the original for a single neighbour.
    assert np.alltrue(x_aug == X_neighs)

    # What happens if a single weight is selected for multiple neighbours.
    W = [1, 0, 0]
    X_neighs = X0[0:3, :]
    Y_neighs = Y0[0:3, :]
    x_aug, y_aug = combine_neighbours(W, X_neighs, Y_neighs)

    # The augmented should be the same as the original one with all the weight on it.
    assert np.alltrue(x_aug == X_neighs[0])
    assert np.alltrue(Y_neighs[0] == y_aug)