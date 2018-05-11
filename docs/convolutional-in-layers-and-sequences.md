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

I've found, recently, that the Sequential class in Keras and PyTorch are very similar to the Layer or Layers APIs in CNTK and TensorFlow - perhaps Sequential is a little bit higher-level so it depends on how much customizability you want, as usual, in these cases.  Below you will see examples of similar CNN architectures in the four different frameworks.

### Keras

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def convNetKeras(num_classes):
    """ Adapted from: http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
    """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))

    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay,
         nesterov=False)
    model.compile(loss='categorical_crossentropy',
         optimizer=sgd, metrics=['accuracy'])
    return model
```

What you don't see is:
* Fit/train (`model.fit()`)
* Evaluate with given metric (`model.evaluate()`)
* To add dropout after the `Convolution2D()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g., `Dropout(0.5)`

### PyTorch

(CNN)

```python
import torch
import torch.nn as nn

class ConvNetPyTorch(nn.Module):
    """Adapted from:
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.5)
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNetPyTorch(num_classes).to(device)
```

What you don't see is:
* Fit/train (`model.train()`)
* Evaluate with given metric (`model.eval()`)
* To add dropout after the `nn.ReLU()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `nn.Dropout(0.5)`

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
* To add dropout after the `C.layers.Convolution()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `C.layers.Dropout(0.5)`

## Conclusion

## References

1.  Kaggle Keras code sample [Ref](https://www.kaggle.com/tonypoe/keras-cnn-example?scriptVersionId=589403)
2.  PyTorch code sample [Ref](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py)
3.  CNTK code sample with Layer API [Doc](https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html)
4.  TensorFlow code sample with Layers API [Doc](https://www.tensorflow.org/tutorials/layers) and ConvNets Tutorial at this [Doc](https://www.tensorflow.org/tutorials/deep_cnn)

* Keras example:  http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
* PyTorch example:  https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
* TensorFlow example: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
* CNTK example:  https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html 

Thanks for reading.