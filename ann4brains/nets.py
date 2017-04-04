from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
import cPickle
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from layers import e2e_conv, e2n_conv, full_connect
from optimizers import caffe_SGD
from utils import h5_utils
# import abc
# import six
from utils.h5_utils import caffe_write_h5
from utils.metrics import regression_metrics


def load_model(filepath):
    """Load the model.

    :param filepath: Filename and path to load the model:
    :return: a BrainnetCNN/BaseNet object
    """

    with open(filepath, 'rb') as fid:
        model = cPickle.load(fid)

    model.load_parameters()

    return model


# ABC for python 2 and 3. Now removed to make dependencies easier.
# http://stackoverflow.com/a/35673504/754920
# @six.add_metaclass(abc.ABCMeta)
class BaseNet(object):

    def __init__(self, net_name,
                 arch,  # Neural network architecture.
                 hdf5_train=None, hdf5_validate=None,
                 hardware='gpu',  # Either 'gpu' or 'cpu'
                 dir_data='./generated_synthetic_data',  # Where to store the data.
                 ):
        """Initialize the neural network.

        net_name : (string) unique name to identify the model.
        arch : (dict) specifies the architecture of the network.
        """

        self.net_name = net_name
        # Dictionary stores all the parameters for the optimization.
        self.pars = self.get_default_hyper_params(self.net_name)
        self.arch = arch
        self.hardware = hardware
        self.dir_data = dir_data

        # Iteration number to load the weights (set to None since no weights now).
        self.parameter_iter = None

        if hdf5_train is not None:
            self.hdf5_train = hdf5_train
            self.hdf5_validate = hdf5_validate
            # Load the data to infer some of the parameters (otherwise you'll have to set them in self.pars)
            data = h5_utils.read_h5(self.hdf5_validate, ['data', 'label'])
            self.set_data_dependent_pars(data['data'])

    def __getstate__(self):

        # Some things cannot (or we do not want them to) be pickled.
        # This is a way of specifying what we can pickle.
        # http://stackoverflow.com/questions/2999638/how-to-stop-attributes-from-being-pickled-in-python?noredirect=1&lq=1
        d = dict(self.__dict__)
        del d['net']
        return d

    def set_data_dependent_pars(self, data):
        """Set parameters that are dependent on the data."""

        # Assumes dimensions are [num_samples, num_channels, spatial, spatial].
        data_dims = data.shape

        # Assumes dimensions are [batch_size, num_channels, spatial, spatial].
        self.pars['deploy_dims'] = [1, data_dims[1], data_dims[2], data_dims[3]]

        # The number of test samples should equal, test_batch_size * test_iter.
        # Easiest to set pars['test_iter'] to equal the number of test samples pars['test_batch_size'] = 1
        self.pars['test_iter'] = data_dims[0]

    def create_architecture(self, mode, hdf5_data):
        """Returns the architecture (i.e., caffe prototxt) of the model.

        Jer: One day this should probably be written to be more general.
        """

        arch = self.arch
        pars = self.pars
        n = caffe.NetSpec()

        if mode == 'deploy':
            n.data = L.DummyData(shape=[dict(dim=pars['deploy_dims'])])
        elif mode == 'train':
            n.data, n.label = L.HDF5Data(batch_size=pars['train_batch_size'], source=hdf5_data, ntop=pars['ntop'])
        else:  # Test.
            n.data, n.label = L.HDF5Data(batch_size=pars['test_batch_size'], source=hdf5_data, ntop=pars['ntop'])

        # print(n.to_proto())
        in_layer = n.data

        for layer in arch:
            layer_type, vals = layer

            if layer_type == 'e2e':
                in_layer = n.e2e = e2e_conv(in_layer, vals['n_filters'], vals['kernel_h'], vals['kernel_w'])
            elif layer_type == 'e2n':
                in_layer = n.e2n = e2n_conv(in_layer, vals['n_filters'], vals['kernel_h'], vals['kernel_w'])
            elif layer_type == 'fc':
                in_layer = n.fc = full_connect(in_layer, vals['n_filters'])
            elif layer_type == 'out':
                n.out = full_connect(in_layer, vals['n_filters'])
                # Rename to user specified unique layer name.
                # n.__setattr__('out', n.new_layer)

            elif layer_type == 'dropout':
                in_layer = n.dropout = L.Dropout(in_layer, in_place=True,
                                                 dropout_param=dict(dropout_ratio=vals['dropout_ratio']))
            elif layer_type == 'relu':
                in_layer = n.relu = L.ReLU(in_layer, in_place=True,
                                           relu_param=dict(negative_slope=vals['negative_slope']))
            else:
                raise ValueError('Unknown layer type: ' + str(layer_type))

        # ~ end for.

        if mode != 'deploy':
            if self.pars['loss'] == 'EuclideanLoss':
                n.loss = L.EuclideanLoss(n.out, n.label)
            else:
                ValueError("Only 'EuclideanLoss' currently implemented for pars['loss']!")

        return n

    def create_prototxts(self, hdf5_train, hdf5_validation):  # TODO: Make protected
        """Create and write to disk the prototxt files.

        dir_proto : (string) directory where to write the prototxt files.
        hdf5_train : (string) textfile containing the filename of the training hdf5 dataset.
        hdf5_test : (string) textfile that contains the filename of the testing hdf5 dataset.
        dir_snapshots : (string) directory where to write the trained models.
        """
        pars = self.pars
        net_name = self.net_name
        dir_proto = self.pars['dir_caffe_proto']

        # Create training architecture.
        n = self.create_architecture('train', hdf5_train)
        # Write to disk.
        self.train_proto = os.path.join(dir_proto, net_name + '_train.prototxt')
        with open(self.train_proto, "w") as f:
            f.write(str(n.to_proto()))

        # Validation architecture.
        n = self.create_architecture('test', hdf5_validation)
        self.test_proto = os.path.join(dir_proto, net_name + '_test.prototxt')
        with open(self.test_proto, "w") as f:
            f.write(str(n.to_proto()))

        # Testing (deploy) architecture.
        n = self.create_architecture('deploy', None)
        self.deploy_proto = os.path.join(dir_proto, net_name + '_deploy.prototxt')
        with open(self.deploy_proto, "w") as f:
            f.write(str(n.to_proto()))

        # Solver.
        s = self.create_caffe_solver(os.path.join(self.pars['dir_snapshots'], net_name))
        self.solver_proto = os.path.join(dir_proto, net_name + '_solver.prototxt')
        with open(self.solver_proto, "w") as f:
            f.write(str(s))

    def fit(self, x_train=None, y_train=None, x_valid=None, y_valid=None):
        """Train the model.

        For caffe, assumes that self.hdf5_train/validate have already been set.
        So X,y are not necessary.
        We keep the X,y to match the keras/sklearn api consistent and for future extensions that take in X,y.
        """
        if self.pars['dl_framework'] == 'caffe':
            if x_train is not None:
                # We are getting the raw data, so write the data to disk in a h5 format for caffe.
                self.hdf5_train = os.path.abspath(os.path.join(self.dir_data, 'train.h5'))
                self.hdf5_validate = os.path.abspath(os.path.join(self.dir_data, 'valid.h5'))
                caffe_write_h5(self.hdf5_train, x_train, y_train)
                caffe_write_h5(self.hdf5_validate, x_valid, y_valid)
                self.set_data_dependent_pars(x_valid)

            # Create the prototxt files
            self.create_prototxts(self.hdf5_train + '.txt', self.hdf5_validate + '.txt')

            # Optimize the net.
            self.train_metrics, self.test_metrics = caffe_SGD(self.solver_proto, self.pars['max_iter'],
                                                              self.pars['test_interval'], self.pars['test_iter'],
                                                              compute_metrics_func=self.pars['compute_metrics_func'],
                                                              start_weights_name=None, set_mode=self.hardware)

            # Then load the parameters from the last iteration.
            self.parameter_iter = self.pars['max_iter']
            self.load_parameters()
        else:
            print('No valid deep learning framework specified in hyper parameter \'dl_framework\'')

    # def plot_error(self):
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(1, 1, 1)
    #    plot_err_iter(ax1, self.pars['net_name'], self.train_metrics[np.newaxis, :], self.test_metrics[np.newaxis, :],
    #                  self.pars['max_iter'], self.pars['test_interval'])

    def load_parameters(self, iter_num=None):
        """Load the trained weights/parameters at iteration iter_num.

        :param iter_num: the iteration at which to load the weights.
        """
        # Report results on the last trained model.

        if iter_num is None:
            # Load the parameters  trained model.
            iter_num = self.parameter_iter

        snapshot_caffemodel = os.path.join(self.pars['dir_snapshots'],
                                           self.net_name + '_iter_' + str(iter_num) + '.caffemodel')
        self.net = caffe.Net(self.deploy_proto, snapshot_caffemodel, caffe.TEST)

    def save(self, filepath):
        """Save the object (model, weights).

        :param filepath: name and path of where to save the model.
        """
        with open(filepath, 'wb') as fid:
            cPickle.dump(self, fid)

    # @abc.abstractmethod
    def predict(self, X):
        """Computes the predictions for X using the trained model."""
        # Implement in the inherited class.
        preds = []
        return preds


    # @abc.abstractmethod
    @staticmethod
    def print_results(X, Y):
        """Displays the results"""
        # Implement in the inherited class.
        pass

    # @abc.abstractmethod
    def plot_iter_metrics(self):
        """Plot the train, test metrics over iterations."""
        # Implement in the inherited class.
        pass

    def caffe_get_layer_response(self, net, preprocessed_x, response_layer_id='out', input_layer_id='data'):
        """Returns the responses at a specified layer for the given input."""

        net.blobs[input_layer_id].data[...] = preprocessed_x
        output = net.forward()
        responses = np.copy(net.blobs[response_layer_id].data)

        return responses

    def create_caffe_solver(self, snapshot_prefix):
        s = caffe_pb2.SolverParameter()

        s.train_net = self.train_proto
        if self.test_proto is not None:
            s.test_net.append(self.test_proto)
            # Test every 'test_interval' iterations.
            s.test_interval = self.pars['test_interval']
            # Batch size to test. IMPORTANT! test_iter * test_proto.batch_size should equal your test data.
            s.test_iter.append(self.pars['test_iter'])

        # Learning rate.
        s.base_lr = self.pars['base_learning_rate']
        s.lr_policy = 'step'
        s.gamma = 0.1  # CJB: whats this? Should we create a hyper parameter for it?
        s.stepsize = self.pars['step_size']
        s.max_iter = self.pars['max_iter']
        s.display = self.pars['train_interval']
        s.momentum = self.pars['learning_momentum']
        s.weight_decay = self.pars['weight_decay']
        s.snapshot = self.pars['snapshot']
        s.snapshot_prefix = snapshot_prefix
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        s.random_seed = 333

        return s

    def _get_mat_preds(self, net, X, input_layer_id='data', response_layer_id='out'):
        if self.pars['dl_framework'] == 'caffe':
            preds = []
            for x in X:
                p = self.caffe_get_layer_response(net, x, response_layer_id, input_layer_id)
                preds.append(np.squeeze(p))
        else:
            print('No valid deep learning framework specified in hyper parameter \'dl_framework\'')

        return np.asarray(preds)

    @staticmethod
    def get_default_hyper_params(net_name):
        """Return a dict of the default neural network hyper-parameters"""

        pars = {}
        pars['net_name'] = net_name  # A unique name of the model.
        pars['dl_framework'] = 'caffe'  # To use different backend neural network frameworks (only caffe for now).

        # Solver parameters
        pars['train_interval'] = 100  # Display the loss over the training data after 100 iterations.
        pars['test_interval'] = 100  # Check the model over the test/validation data after this many iterations.
        pars['max_iter'] = 2000  # Max number of iterations to train the model for.
        pars['snapshot'] = 1000  # After how many iterations should we save the model.
        pars['base_learning_rate'] = 0.01  # Initial learning rate.
        pars['step_size'] = 100000  # After how many iterations should we decrease the learning rate.
        pars['learning_momentum'] = 0.9  # Momentum used in learning.
        pars['weight_decay'] = 0.0005  # Weight decay penalty.

        pars['loss'] = 'EuclideanLoss'  # The loss to use (currently only EuclideanLoss works)
        pars['compute_metrics_func'] = regression_metrics  # Can override with your own as long as follows format.

        # Network parameters
        pars['train_batch_size'] = 14  # Size of the training mini-batch.

        # The number of test samples should equal, test_batch_size * test_iter.
        # Easiest to set pars['test_iter'] to equal the number of test samples pars['test_batch_size'] = 1
        pars['test_batch_size'] = 1  # Size of the testing mini-batch.
        # pars['test_iter'] = 56  # How many test samples we have (we'll set this based on the data)

        pars['ntop'] = 2  # How many outputs for the hdf5 data layer (e.g., 'data', 'label' = 2)

        pars['dir_snapshots'] = './snapshot'  # Where to store the trained models
        pars['dir_caffe_proto'] = './proto'  # Where to store the caffe prototxt files.

        return pars


class BrainNetCNN(BaseNet):

    def predict(self, X):
        """Computes the predictions for X.

        X : (4D numpy array) of size N x C x H x W, where N is the number of samples, C is
                the number of channels in each sample, and, H and W are the spatial dimensions
                for each sample.
        """
        preds = self._get_mat_preds(self.net, X, input_layer_id='data', response_layer_id='out')
        return preds

    @staticmethod
    def print_results(preds, y_test):
        """Print the results for this experiment.

        This assumes two classes. And that utils.metrics.regression_metrics computes the desired metrics.
        You might have to override this if your problem is different.

        :param preds: the predicted values.
        :param y_test: the true labels.
        """

        print('E2E prediction results')
        test_metrics_0 = regression_metrics(preds[:, 0], y_test[:, 0])
        print('%s => mae: %.3f, SDAE: %0.3f, corr: %.3f, p-val: %.3f' % ('class 0',
                                                                         test_metrics_0['mad'],
                                                                         test_metrics_0['std_mad'],
                                                                         test_metrics_0['corr_0'],
                                                                         test_metrics_0['p_0']))

        test_metrics_1 = regression_metrics(preds[:, 1], y_test[:, 1])
        print('%s => mae: %.3f, SDAE: %0.3f, corr: %.3f, p-val: %.3f' % ('class 1',
                                                                         test_metrics_1['mad'],
                                                                         test_metrics_1['std_mad'],
                                                                         test_metrics_1['corr_0'],
                                                                         test_metrics_1['p_0']))

    def plot_iter_metrics(self):
        """Plot the train, test metrics over iterations.

        This assumes two classes. And that utils.metrics.regression_metrics was used to monitor the performance.
        You might have to override this if your problem is different.
        """

        itr = []
        # met_keys = test_metrics[0][1].keys()
        # mets = [[]]*len(met_keys)
        mad = []
        corr_0 = []
        corr_1 = []

        for met in self.test_metrics:
            itr.append(met[0])
            # for m_key in met_keys:
            #    mets[0].append(met[1][m_key])
            mad.append(met[1]['mad'])
            corr_0.append(met[1]['corr_0'])
            corr_1.append(met[1]['corr_1'])

        fig, ax = plt.subplots()
        axes = [ax, ax.twinx()]
        axes[0].plot(self.train_metrics, color='purple', label='train loss')
        axes[0].plot(itr, mad, color='red', label='valid mad')
        axes[1].plot(itr, corr_0, color='blue', label='valid corr_0')
        axes[1].plot(itr, corr_1, color='green', label='valid corr_1')

        lines, labels = axes[0].get_legend_handles_labels()
        lines2, labels2 = axes[1].get_legend_handles_labels()
        axes[1].legend(lines + lines2, labels + labels2, loc='best')
