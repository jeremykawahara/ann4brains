# ann4brains

ann4brains implements filters for adjacency matrices that can be used within a deep neural network. These filters are designed specifically for brain network connectomes, but could be used with any adjacency matrix. 

ann4brains is a Python wrapper for [Caffe](https://github.com/BVLC/caffe) that implements the Edge-to-Edge, and Edge-to-Node filters as described in:

> Kawahara, J., Brown, C. J., Miller, S. P., Booth, B. G., Chau, V., Grunau, R. E., Zwicker, J. G., and Hamarneh, G. (2017). BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment. NeuroImage, 146(July), 1038–1049. [[DOI]](https://doi.org/10.1016/j.neuroimage.2016.09.046) [[URL]](http://brainnetcnn.cs.sfu.ca/) [[PDF]](http://www.cs.sfu.ca/~hamarneh/ecopy/neuroimage2016.pdf)

To run an example program on synthetic data, 
```
git clone https://github.com/jeremykawahara/ann4brains.git
jupyter notebook ann4brains/examples/brainnetcnn.ipynb
```

Here's a fully working, minimal ["hello world" example here](https://github.com/jeremykawahara/ann4brains/blob/master/examples/helloworld.ipynb),

```python
import os
import numpy as np
import sys
from scipy.stats.stats import pearsonr
import caffe

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import ann4brains
from ann4brains.synthetic.injury import ConnectomeInjury
from ann4brains.nets import BrainNetCNN

np.random.seed(seed=333) # To reproduce results.

# Generate train/test synthetic data.
injury = ConnectomeInjury()
x_train, y_train = injury.generate_injury()
x_test, y_test = injury.generate_injury()
x_valid, y_valid = injury.generate_injury()

# Unique name for the model
net_name = 'hello_world'

# We specify the architecture like this.
hello_arch = [
    ['e2n', {'num_output': 16,  # e2n layer with 16 filters.
             'kernel_h': x_train.shape[2], 
             'kernel_w': x_train.shape[3]}], # Same dimensions as spatial inputs.
    ['dropout', {'dropout_ratio': 0.5}], # Dropout at 0.5
    ['relu',    {'negative_slope': 0.33}], # For leaky-ReLU
    ['fc',      {'num_output': 30}],  # Fully connected (n2g) layer with 30 filters.
    ['relu',    {'negative_slope': 0.33}],
    ['out',     {'num_output': 1}]  # Output layer with num_outs nodes as outputs.
]

# Create BrainNetCNN model
hello_net = BrainNetCNN(net_name, hello_arch)

# Train the network (the synthetic data has 2 classes, but we use just the first class)
hello_net.fit(x_train, y_train[:,0], x_valid, y_valid[:,0])

# Predict labels of test data
preds = hello_net.predict(x_test)

print("Mean Absolute Difference:" , np.mean(np.abs(preds-y_test[:,0])))
print("Correlation:", pearsonr(preds, y_test[:,0])[0])
```

More examples can be found in this [extended notebook](https://github.com/jeremykawahara/ann4brains/blob/master/examples/brainnetcnn.ipynb).

If you prefer to work directly with [Caffe](https://github.com/BVLC/caffe) and not use this wrapper, you can modify the [example prototxt files](https://github.com/jeremykawahara/ann4brains/tree/master/examples/proto) that implement the E2E and E2N filters.
