# ann4brains

ann4brains implements filters for adjacency matrices that can be used within a deep neural network. These filters are designed specifically for brain network connectomes, but could be used with any adjacency matrix. 

ann4brains is a Python wrapper for [Caffe](https://github.com/BVLC/caffe) that implements the Edge-to-Edge, and Edge-to-Node filters as described in:

> Kawahara, J., Brown, C. J., Miller, S. P., Booth, B. G., Chau, V., Grunau, R. E., Zwicker, J. G., and Hamarneh, G. (2017). BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment. NeuroImage, 146(July), 1038–1049. [[DOI]](https://doi.org/10.1016/j.neuroimage.2016.09.046) [[URL]](http://brainnetcnn.cs.sfu.ca/) [[PDF]](http://www.cs.sfu.ca/~hamarneh/ecopy/neuroimage2016.pdf)

To run an example program on synthetic data, 
```
git clone https://github.com/jeremykawahara/ann4brains.git
jupyter notebook ann4brains/examples/brainnetcnn.ipynb
```

The core part of the code is this,

```python
from ann4brains.nets import BrainNetCNN

# We specify the architecture like this.
e2n_arch = [
    ['e2n', {'num_output': 16,  # e2n layer with 16 filters.
             'kernel_h': d, 'kernel_w': d}],  # Same dimensions as spatial inputs.
    ['dropout', {'dropout_ratio': 0.5}],
    ['relu',    {'negative_slope': 0.33}],
    ['fc',      {'num_output': 30}],  # fully connected (n2g) layer with 30 filters.
    ['relu',    {'negative_slope': 0.33}],
    ['out',     {'num_output': n_injuries}]  # output layer with num_outs nodes as outputs.
]

# Create BrainNetCNN model
hdf5_train = 'train_data.h5'
hdf5_validate = 'validate_data.h5'
E2Nnet_sml = BrainNetCNN('E2Nnet', e2n_arch, hdf5_train, hdf5_validate)

# Train the network.
E2Nnet_sml.fit(set_mode='cpu') # or set_mode='gpu'

# Predict labels of test data
preds = E2Nnet_sml.predict(test_data['data'])
```

If you prefer to work directly with [Caffe](https://github.com/BVLC/caffe) and not use this wrapper, you can modify the [example prototxt files](https://github.com/jeremykawahara/ann4brains/tree/master/examples/proto) that implement the E2E and E2N filters.
