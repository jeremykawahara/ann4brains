from __future__ import print_function
import numpy as np
import os
import caffe
from caffe.proto import caffe_pb2
from caffe import layers as L
from layers import e2e_conv, e2n_conv, full_connect
from optimizers import caffe_SGD
from utils import h5_utils
import abc
import six
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


# ABC for python 2 and 3
# http://stackoverflow.com/a/35673504/754920
@six.add_metaclass(abc.ABCMeta)
class BaseNet(object):

    def __init__(self, net_name, arch, hdf5_train="", hdf5_validate=""):
        """Initialize the neural network.

        net_name : (string) unique name to identify the model.
        arch : (dict) specifies the architecture of the network.
        """

        self.net_name = net_name
        # Dictionary stores all the parameters for the optimization.
        self.pars = self.get_default_hyper_params(self.net_name)
        self.arch = arch

        self.hdf5_train = hdf5_train
        self.hdf5_validate = hdf5_validate

        # Load the validate data to infer some of the parameters (otherwise you'll have to set them in self.pars)
        validate_data = h5_utils.read_h5(self.hdf5_validate, ['data', 'label'])
        # Assumes dimensions are [num_samples, num_channels, spatial, spatial].
        data_dims = validate_data['data'].shape

        # Assumes dimensions are [batch_size, num_channels, spatial, spatial].
        self.pars['deploy_dims'] = [1, data_dims[1], data_dims[2], data_dims[3]]

        # The number of test samples should equal, test_batch_size * test_iter.
        # Easiest to set pars['test_iter'] to equal the number of test samples pars['test_batch_size'] = 1
        self.pars['test_iter'] = data_dims[0]

    @abc.abstractmethod
    def create_architecture(self, mode, hdf5_data):
        """Return the architecture for the net. Implement in the inherited class."""
        n = []
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

    def fit(self, X=None, y=None, set_mode='gpu'):
        """Train the model.

        For caffe, assumes that self.hdf5_train/validate have already been set.
        So X,y are not necessary.
        We keep the X,y to match the keras/sklearn api consistent and for future extensions that take in X,y.
        """
        if self.pars['dl_framework'] == 'caffe':
            # Create the prototxt files
            self.create_prototxts(self.hdf5_train + '.txt', self.hdf5_validate + '.txt')

            # Optimize the net.
            self.train_metrics, self.test_metrics, self.preds, self.actuals = caffe_SGD(self.solver_proto,
                                                                                        self.pars['max_iter'],
                                                                                        self.pars['test_interval'],
                                                                                        self.pars['test_iter'],
                                                                                        start_weights_name=None,
                                                                                        set_mode=set_mode)

            # Then load the parameters for the last iteration.
            self.load_net(self.pars['max_iter'])
        else:
            print('No valid deep learning framework specified in hyper parameter \'dl_framework\'')

    #def plot_error(self):
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(1, 1, 1)
    #    plot_err_iter(ax1, self.pars['net_name'], self.train_metrics[np.newaxis, :], self.test_metrics[np.newaxis, :],
    #                  self.pars['max_iter'], self.pars['test_interval'])

    def load_net(self, iter_num=None):
        # Report results on the last trained model.

        if iter_num is None:
            # Load the parameters for the last trained model.
            iter_num = self.pars['max_iter']

        snapshot_caffemodel = os.path.join(self.pars['dir_snapshots'],
                                           self.net_name + '_iter_' + str(iter_num) + '.caffemodel')
        self.net = caffe.Net(self.deploy_proto, snapshot_caffemodel, caffe.TEST)

    @abc.abstractmethod
    def predict(self, X):
        """Computes the predictions for X using the trained model."""
        preds = []
        return preds

    @staticmethod
    @abc.abstractmethod
    def print_results(X, Y):
        """Displays the results"""
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
        s.momentum = self.pars['learning_momentum']
        s.weight_decay = self.pars['weight_decay']
        s.snapshot = self.pars['snapshot']
        s.snapshot_prefix = snapshot_prefix
        s.solver_mode = caffe_pb2.SolverParameter.GPU
        s.random_seed = 333

        return s

    @staticmethod
    def get_default_hyper_params(net_name):
        """Return a dict of the default neural network hyper-parameters"""

        pars = {}
        pars['net_name'] = net_name
        pars['dl_framework'] = 'caffe'

        # Solver parameters
        pars['test_interval'] = 500  # Check the model over the test/validation data after this many iterations.
        pars['max_iter'] = 100000  # Max number of iterations to train the model for.
        pars['snapshot'] = 10000  # After how many iterations should we save the model.
        pars['base_learning_rate'] = 0.01
        pars['step_size'] = 100000
        pars['learning_momentum'] = 0.9
        pars['weight_decay'] = 0.0005

        # Network parameters
        pars['train_batch_size'] = 14
        pars['test_batch_size'] = 1
        pars['ntop'] = 2  # How many outputs for the hdf5 data layer (e.g., 'data', 'label' = 2)

        pars['dir_snapshots'] = './snapshot'  # Where to store the trained models
        pars['dir_caffe_proto'] = './proto'  # Where to store the caffe prototxt files.

        return pars


class BrainNetCNN(BaseNet):
    def create_architecture(self, mode, hdf5_data):
        """Returns the architecture (i.e., caffe prototxt) of the model.

        Jer: One day this should probably be written to be more general and go into the BaseNet.
        """

        arch = self.arch
        pars = self.pars
        n = caffe.NetSpec()

        if mode == 'deploy':
            n.data = L.DummyData(shape=dict(dim=pars['deploy_dims']))
        elif mode == 'train':
            n.data, n.label = L.HDF5Data(batch_size=pars['train_batch_size'], source=hdf5_data, ntop=pars['ntop'])
        else:  # Test.
            n.data, n.label = L.HDF5Data(batch_size=pars['test_batch_size'], source=hdf5_data, ntop=pars['ntop'])

        # print(n.to_proto())
        in_layer = n.data

        for layer in arch:
            layer_type, vals = layer

            if layer_type == 'e2e':
                in_layer = n.e2e = e2e_conv(in_layer, vals['num_output'], vals['kernel_h'], vals['kernel_w'])
            elif layer_type == 'e2n':
                in_layer = n.e2n = e2n_conv(in_layer, vals['num_output'], vals['kernel_h'], vals['kernel_w'])
            elif layer_type == 'fc':
                in_layer = n.fc = full_connect(in_layer, vals['num_output'])
            elif layer_type == 'out':
                n.out = full_connect(in_layer, vals['num_output'])
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
            n.loss = L.EuclideanLoss(n.out, n.label)

        return n

    def get_mat_preds(self, net, X, input_layer_id='data', response_layer_id='out'):
        if self.pars['dl_framework'] == 'caffe':
            preds = []
            for x in X:
                p = self.caffe_get_layer_response(net, x, response_layer_id, input_layer_id)
                preds.append(np.squeeze(p))
        else:
            print('No valid deep learning framework specified in hyper parameter \'dl_framework\'')

        return np.asarray(preds)

    def predict(self, X):
        """Computes the predictions for X.

        X : (4D numpy array) of size N x C x H x W, where N is the number of samples, C is
                the number of channels in each sample, and, H and W are the spatial dimensions
                for each sample.
        """
        preds = self.get_mat_preds(self.net, X, input_layer_id='data', response_layer_id='out')
        return preds

    @staticmethod
    def compute_prediction_metrics(pred_values, actual_values):
        """Returns the metrics for predicted and true values. Assumes a 1D prediction."""

        madErr = np.mean(abs(pred_values - actual_values))
        std_mad = np.std(abs(pred_values - actual_values))
        c, p = pearsonr(pred_values, actual_values)

        return madErr, std_mad, c, p

    @staticmethod
    def print_results(X, Y):
        class_idx = 0
        cog_madErr, cog_mad_std, cog_c, cog_p = BrainNetCNN.compute_prediction_metrics(X[:, class_idx], Y[:, class_idx])
        print('%s => mae: %.4f, corr: %.4f, p-val: %.4f' % ('rho', cog_madErr, cog_c, cog_p))

        class_idx = 1
        mot_madErr, mot_mad_std, mot_c, mot_p = BrainNetCNN.compute_prediction_metrics(X[:, class_idx], Y[:, class_idx])
        print('%s => mae: %.4f, corr: %.4f, p-val: %.4f' % ('rho', mot_madErr, mot_c, mot_p))

        # print '& %.3f & %.4f & %.3f & %.3f & %.3f & %.4f & %.3f & %.3f \\\\' % (mot_c, mot_p, mot_madErr*100, mot_mad_std*100, cog_c, cog_p, cog_madErr*100, cog_mad_std*100)

        print(mot_c + cog_c)