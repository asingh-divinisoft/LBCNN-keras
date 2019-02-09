# Imports
from scipy.stats import bernoulli
from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D
from keras import backend as K
from keras.engine.topology import Layer

# Non-trainable filters initialized with distribution
# of Bernoulli as in article and then it's non-trainable
def new_weights_non_trainable(h, w, num_input, num_output, sparsity=0.5):
    # Number of elements
    num_elements = h * w * num_input * num_output
    # Create an array with n number of elements
    array = np.arange(num_elements)
    # Random shuffle it
    np.random.shuffle(array)
    # Fill with 0
    weight = np.zeros([num_elements])
    # Get number of elements in array that need be non-zero
    ind = int(sparsity * num_elements + 0.5)
    # Get it piece as indexes for weight matrix
    index = array[:ind]
  
    for i in index:
        # Fill those indexes with bernoulli distribution
        # Method rvs = random variates
        weight[i] = bernoulli.rvs(0.5)*2-1
    # Reshape weights array for matrix that we need
    weights = weight.reshape(h, w, num_input, num_output)
    return weights

class LBC(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding='same', activation='relu', dilation=1, sparsity=0.9, name=None):
        super(LBC, self).__init__()
        self.nOutputPlane = filters
        self.kW = kernel_size
        self.sparsity = sparsity
        self.LBC = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding=padding,
                            dilation_rate=dilation, activation=activation, use_bias=False, name=name)

    def build(self, input_shape):
        nInputPlane = input_shape[-1]
        
        with K.name_scope(self.LBC.name):
            self.LBC.build(input_shape)
        # Create a trainable weight variable for this layer.
        anchor_weights = tf.Variable(new_weights_non_trainable(h=self.kW, w=self.kW, num_input=nInputPlane,
                                                           num_output=self.nOutputPlane, sparsity=self.sparsity).astype(np.float32),
                                                           trainable=False)
        #self.LBC._trainable_weights = []
        self.LBC.kernel = anchor_weights
        #self.LBC._non_trainable_weights.append(anchor_weights)
        #self.trainable = False
        super(LBC, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.LBC(x)

    def compute_output_shape(self, input_shape):
        return self.LBC.compute_output_shape(input_shape)
