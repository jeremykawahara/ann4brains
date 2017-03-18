import caffe
import numpy as np
from scipy.stats.stats import pearsonr


def get_error_measures(pred_labels, true_labels):
    # print pred_labels.shape
    # print true_labels.shape

    if np.shape(np.squeeze(pred_labels).shape)[0] > 1:
        if pred_labels.shape[1] > 2:
            pred_values = pred_labels[:, 0]
            actual_values = true_labels[:, 0]
            r_0, p = pearsonr(pred_values, actual_values)

            pred_values = pred_labels[:, 1]
            actual_values = true_labels[:, 1]
            r_1, p = pearsonr(pred_values, actual_values)

            pred_values = pred_labels[:, 2]
            actual_values = true_labels[:, 2]
            r_2, p = pearsonr(pred_values, actual_values)

            mae_dist = np.mean((np.sum((abs(pred_labels - true_labels)), axis=1)))

            # c = np.asarray([c_0,c_1,c_2])
            return mae_dist, r_0, r_1, r_2

        else:  # np.shape(np.squeeze(pred_labels).shape)[0] > 1:
            pred_values = pred_labels[:, 0]
            actual_values = true_labels[:, 0]
            r_0, p = pearsonr(pred_values, actual_values)

            pred_values = pred_labels[:, 1]
            actual_values = true_labels[:, 1]
            r_1, p = pearsonr(pred_values, actual_values)

            mae_dist = np.mean((np.sum((abs(pred_labels - true_labels)), axis=1)))

            return mae_dist, r_0, r_1,

    else:
        r_0, p = pearsonr(pred_labels, true_labels)
        mae_dist = np.mean((np.sum((abs(pred_labels - true_labels)), axis=1)))

    return mae_dist, r_0


def caffe_SGD(solver_filename, niter, test_interval, test_iter, start_weights_name=None, set_mode='gpu',
              pred_layer_id='out'):
    '''
    Runs the caffe stochastic gradient descent solver on the solver_filename for niter.

    Input:
    solver_filename - a string containing the path and filename of the solver.prototxt to load
    niter - number of gradient steps to take.
    test_interval - number of gradient steps to take before testing.
    test_iter - how many batches needed to pass over the entire test data (test_iter *batch_size = test_data)
    '''

    # Load a new solver each time.
    solver = caffe.SGDSolver(solver_filename)

    num_metrics = solver.net.blobs[pred_layer_id].data.shape[1] + 1

    if start_weights_name != None:
        print 'starting from: ' + start_weights_name
        solver.restore(start_weights_name)
        # solver.net.copy_from(start_weights_name)

    if set_mode == 'cpu':
        caffe.set_mode_cpu()
    elif set_mode == 'gpu':
        # print "CANCELED GPU!!!"
        caffe.set_mode_gpu()
        # caffe.set_device(GPU_ID)
    else:
        display('error: invalid set_mode value')

    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_acc = np.zeros(shape=(int(np.ceil(niter / test_interval)), num_metrics))

    # the main solver loop
    for it in range(niter):

        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            # print 'Iteration', it, 'train-loss', train_loss[it], 'testing...'
            preds = np.asarray([])
            actuals = np.asarray([])
            for test_idx in range(test_iter):
                solver.test_nets[0].forward()
                pred = np.copy(solver.test_nets[0].blobs[pred_layer_id].data)
                actual = np.copy(solver.test_nets[0].blobs['label'].data)
                preds = np.vstack([preds, pred]) if preds.size else pred
                actuals = np.vstack([actuals, actual]) if actuals.size else actual

            out = get_error_measures(preds, actuals)  # Compute our own metric.
            test_acc[it // test_interval, :] = out

    # Maybe reuse the solver...?
    return train_loss, test_acc, preds, actuals  # Returns the last set of preds.