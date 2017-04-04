# Software by Jeremy Kawahara and Colin J Brown
# Medical Image Analysis Lab, Simon Fraser University, Canada, 2017
# Simple "Hello World" example.

import os, sys
import numpy as np
from scipy.stats.stats import pearsonr
import caffe
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import ann4brains.
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN
np.random.seed(seed=333) # To reproduce results.

injury = ConnectomeInjury() # Generate train/test synthetic data.
x_train, y_train = injury.generate_injury()
x_test, y_test = injury.generate_injury()
x_valid, y_valid = injury.generate_injury()

hello_arch = [ # We specify the architecture like this.
    ['e2n', {'n_filters': 16,  # e2n layer with 16 filters.
             'kernel_h': x_train.shape[2], 
             'kernel_w': x_train.shape[3]}], # Same dimensions as spatial inputs.
    ['dropout', {'dropout_ratio': 0.5}], # Dropout at 0.5
    ['relu',    {'negative_slope': 0.33}], # For leaky-ReLU
    ['fc',      {'n_filters': 30}],  # Fully connected (n2g) layer with 30 filters.
    ['relu',    {'negative_slope': 0.33}],
    ['out',     {'n_filters': 1}]]  # Output layer with num_outs nodes as outputs.

hello_net = BrainNetCNN('hello_world', hello_arch) # Create BrainNetCNN model
hello_net.fit(x_train, y_train[:,0], x_valid, y_valid[:,0]) # Train (regress only on class 0)
preds = hello_net.predict(x_test) # Predict labels of test data
print("Correlation:", pearsonr(preds, y_test[:,0])[0])

