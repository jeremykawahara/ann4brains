import numpy as np
from scipy.stats.stats import pearsonr


def regression_metrics(pred_labels, true_labels):
    """Return metrics for regression."""
    met = {}

    met['mad'] = np.mean((abs(pred_labels - true_labels)))
    met['std_mad'] = np.std(abs(pred_labels - true_labels))

    if np.shape(np.squeeze(pred_labels).shape)[0] > 1:
        # There's multiple labels.
        n_labels = pred_labels.shape[1]
        for idx in range(n_labels):
            pred_values = pred_labels[:, idx]
            actual_values = true_labels[:, idx]
            r, p = pearsonr(pred_values, actual_values)
            met['corr_' + str(idx)] = r
            met['p_' + str(idx)] = p

    else:  # Only 1 label.
        r, p = pearsonr(pred_labels, true_labels)
        met['corr_0'] = r
        met['p_0'] = p

    return met
