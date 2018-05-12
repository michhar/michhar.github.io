---
img:  IMG_3568.JPG
layout: post
title: Back-Propagation in Deep Learning Frameworks
comments: true
description: A tech piece on back-propagation
cover:  
tags: [python, deep-learning, tensorflow, cntk, keras, pytorch]
---

**tl;dr**:  

**Posted:**  2018-05-12

## Introduction

I've found recently that the Sequential class and Layer/Layers modules are names used across Keras, PyTorch, TensorFlow and CNTK - making it a little confusing to switch from one framework to another.  I was also curious about using these modules/APIs in each framework to define a Convolutional neural network ([ConvNet](https://en.wikipedia.org/wiki/Convolutional_neural_network)).

The neural network archicture here is:

1. Convolutional layer
2. Max pooling layer
3. Convolutional layer
4. Max pooling layer
5. Fully connected or dense layer with 10 outputs and softmax activation (to get probabilities)

> Note: Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer and Dropout layers may be added after the convolutional layers (or pooling) or right before a dense layer to decrease parameter space to help prevent overfitting.

### Keras

```python
"""
Adapted from:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

What you don't see is:
* Fit/train (`model.fit()`)
* Evaluate with given metric (`model.evaluate()`)
* To add dropout after the `Convolution2D()` layer (or after the fully connected in any of these examples) a dropout function will be used, e.g., `Dropout(0.5)`
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

### PyTorch

(CNN)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetPyTorch(nn.Module):
    """Adapted from:
    https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/convnet.ipynb
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 28x28x1 => 28x28x4
            nn.Conv2d(in_channels=1,
                      out_channels=4,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1), # (1(28-1) - 28 + 3) / 2 = 1
            nn.ReLU(),
            # 28x28x4 => 14x14x4
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2),
                         padding=0)) # (2(14-1) - 28 + 2) = 0    
        self.layer2 = nn.Sequential(
            # 14x14x4 => 14x14x8
            nn.Conv2d(in_channels=4,
                      out_channels=8,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1) # (1(14-1) - 14 + 3) / 2 = 1   
            nn.ReLU(),
            # 14x14x8 => 7x7x8 
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2),
                         padding=0)) # (2(7-1) - 14 + 2) = 0
        self.linear_1 = nn.Linear(7*7*8, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        logits = self.linear_1(out.view(-1, 7*7*8))
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = ConvNetPyTorch(num_classes).to(device)
```

What you don't see is:
* Fit/train (`model.train()`)
* Evaluate with given metric (`model.eval()`)
* To add dropout after the `nn.ReLU()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `nn.Dropout(0.5)`
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

### Tensorflow

Below is a CNN defined with the Layers library (with some nice comments!).

```python
import tensorflow as tf

num_classes = 10 # MNIST total classes (0-9 digits)
# Create the neural network
def convNetTensorFlow(x_dict, n_classes, dropout, reuse, is_training):
    """Adapted from:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out
```

What you don't see is:
* Fit/train (`model.train()`)
* Evaluate with given metric (`model.evaluate()`)
* To add dropout after the `tf.layers.conv2d()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `tf.layers.dropout(inputs=net_layer, rate=0.5, training=is_training)`
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

For more see tensorflow in the [References](#references) below.

### Cognitive Toolkit (CNTK)

Below is a CNN defined for MNIST images with Layer API in a long form and a terse form.

```python
def convNetCNTK(input, out_dims):
   """https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html
   """
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        net = C.layers.Convolution((5,5), 32, pad=True)(input)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 32, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Dense(64)(net)
        net = C.layers.Dense(out_dims, activation=None)(net)

    return net
```

The following is the same CNN using the Layer API and the Sequential class to make the code more compact.

```python
def convNetCNTK(input, out_dims):

    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(2), lambda i: [
                C.layers.Convolution((5,5), [32,32][i], pad=True),
                C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
            C.layers.Dense(64),
            C.layers.Dense(out_dims, activation=None)
        ])

    return model(input)
```

What you don't see is:
* Fit/train (`trainer = C.Trainer()` and `trainer.train_minibatch()`)
* Evaluate with given metric (`out = C.softmax()` and `out.eval()`)
* To add dropout after the `C.layers.Convolution()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `C.layers.Dropout(0.5)`.
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

## Conclusion

## References

Samples used in this post:

1.  Keras code sample [Ref](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
2.  PyTorch code sample [Ref](https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/convnet.ipynb)
3.  CNTK code sample with Layer API [Doc](https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html)
4.  TensorFlow code sample with Layers API [Doc](https://www.tensorflow.org/tutorials/layers) and ConvNets Tutorial at this [Doc](https://www.tensorflow.org/tutorials/deep_cnn)


Even more nice code samples:

*  Kaggle Keras code sample [Ref](https://www.kaggle.com/tonypoe/keras-cnn-example?scriptVersionId=589403)
* Keras example:  http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
* PyTorch example: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
* TensorFlow example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
* CNTK example:  https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html 

Thanks for reading.