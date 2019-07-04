import caffe
import numpy as np


def caffe_SGD(solver_filename, niter, test_interval, test_iter,
              compute_metrics_func,
              start_weights_name=None, set_mode='gpu',
              pred_layer_id='out'):
    """
    Runs the caffe stochastic gradient descent solver on the solver_filename for niter.

    Input:
    solver_filename - a string containing the path and filename of the solver.prototxt to load
    niter - number of gradient steps to take.
    test_interval - number of gradient steps to take before testing.
    test_iter - how many batches needed to pass over the entire test data (test_iter *batch_size = test_data)
    compute_metrics_fun: a function that takes pred_labels, and the true_labels, and returns a dict of metrics.
    """

    # Load a new solver each time.
    solver = caffe.get_solver(solver_filename)

    # num_metrics = solver.net.blobs[pred_layer_id].data.shape[1] + 1

    if start_weights_name != None:
        print('starting from: ', start_weights_name)
        solver.restore(start_weights_name)
        # solver.net.copy_from(start_weights_name)

    if set_mode == 'cpu':
        caffe.set_mode_cpu()
    elif set_mode == 'gpu':
        # For some reason, calling caffe.set_mode_gpu() gives the error:
        #
        # Check failed: error == cudaSuccess (33 vs. 0)  invalid resource handle 
        #
        # So best to ignore the GPU flag here.
        # caffe.set_mode_gpu() 
        # caffe.set_device(GPU_ID)
        pass
    else:
        display('error: invalid set_mode value')

    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    # test_acc = np.zeros(shape=(int(np.ceil(niter / test_interval)), num_metrics))
    test_metrics = []

    # the main solver loop
    for it in range(niter):

        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # run a full test every test_interval
        # Caffe can also do this for us and write to a log, but we do directly in python so
        # we can compute our custom metrics.
        if it % test_interval == 0:
            # print 'Iteration', it, 'train-loss', train_loss[it], 'testing...'
            preds = np.asarray([])
            actuals = np.asarray([])
            for test_idx in range(test_iter):
                solver.test_nets[0].forward()
                pred = np.copy(solver.test_nets[0].blobs[pred_layer_id].data)
                actual = np.copy(solver.test_nets[0].blobs['label'].data)
                # Add to the preds/actuals if exist.
                preds = np.vstack([preds, pred]) if preds.size else pred
                actuals = np.vstack([actuals, actual]) if actuals.size else actual

            out = compute_metrics_func(preds, actuals)  # Compute our own metric.
            # test_acc[it // test_interval, :] = out
            test_metrics.append([it, out])

    return train_loss, test_metrics
