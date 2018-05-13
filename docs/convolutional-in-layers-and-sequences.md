---
img:  IMG_3568.JPG
layout: post
title: ConvNets
comments: true
description: A tech piece on back-propagation
cover:  
tags: [python, deep-learning, tensorflow, cntk, keras, pytorch]
---

**tl;dr**:  Un-confusing the naming of current classes and APIs for deep learning frameworks, plus a nice convolutional neural network defined in four deep learning frameworks.

**Posted:**  2018-05-12

![ConvNet Diagram](https://i.stack.imgur.com/ZgG1Z.png)
[Source](http://www.mshahriarinia.com/home/ai/machine-learning/neural-networks/deep-learning/python/theano-mnist/3-convolutional-neural-network-lenet)

## Introduction

I've found recently that the Sequential class and Layer/Layers modules are names used across Keras, PyTorch, TensorFlow and CNTK - making it a little confusing to switch from one framework to another.  I was also curious about using these modules/APIs in each framework to define a Convolutional neural network ([ConvNet](https://en.wikipedia.org/wiki/Convolutional_neural_network)).

Let's get through some terminology.  You can skip to the [Code](#keras) if you are already familiar with ConvNets.

The neural network architecture used in this post is as follows.

1. Convolutional layer
2. Max pooling layer
3. Convolutional layer
4. Max pooling layer
5. Fully connected or dense layer with 10 outputs and softmax activation (to get probabilities)

A convolutional layer creates a feature map (using a _filter_ or _kernel_, which I like to refer to as a "flashlight", shinning on the image and stepping through with a sliding window of 1 unit, that's a _stride_ of 1, by the way).  A good reference for this is in the CNTK [Tutorial](https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html#Convolution-Layer).

![Convolutional layer](https://image.slidesharecdn.com/convnets-151015164458-lva1-app6891/95/deep-learning-convolutional-neural-networks-58-638.jpg?cb=1449100605)
[Source](https://www.slideshare.net/perone/deep-learning-convolutional-neural-networks)


A pooling layer is a way to subsample an input feature map, or output from the convolutional layer that has already done the processing (extracted salient features from) an image in our case.

![Pooling](https://image.slidesharecdn.com/convnets-151015164458-lva1-app6891/95/deep-learning-convolutional-neural-networks-61-638.jpg?cb=1449100605)
[Source](https://www.slideshare.net/perone/deep-learning-convolutional-neural-networks)

Remember, the power of a convolutional layer is that we don't have to do much upfront raw image processing.  The layer(s) will subsequently find the most salient features for us.

A fully connected layer is defined such that every input unit is connected to every output unit much like the [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).

![Dense layer](https://image.slidesharecdn.com/layersintensorflow-170621125437/95/networks-are-like-onions-practical-deep-learning-with-tensorflow-21-638.jpg?cb=1498049767)
[Source](https://www.slideshare.net/barbarafusinska/networks-are-like-onions-practical-deep-learning-with-tensorflow)

Not represented in the code below, but important nonetheless, is dropout.  Dropout removes a percentage of the neuron connections - helping to prevent overfitting by reducing the feature space for convolutional or, especially, dense layers.

![Dropout](https://image.slidesharecdn.com/convnets-151015164458-lva1-app6891/95/deep-learning-convolutional-neural-networks-68-638.jpg?cb=1449100605)
[Source](https://www.slideshare.net/perone/deep-learning-convolutional-neural-networks)

In this post you will find ConvNets defined for four frameworks with adaptations to create easier comparisons (please leave comments as needed).  The full example code can be found as a Jupyter notebook - [Ref](https://github.com/michhar/python-jupyter-notebooks/blob/master/multi_framework/ConvNet_Comparisons.ipynb).

### Keras

Below is a ConvNet defined with the `Sequential` model in Keras ([Ref](https://keras.io/getting-started/sequential-model-guide/)).  This is a snippet with only the model definition parts - see the [References](#references) for the full code example.

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
                 strides=(1, 1),
                 padding='valid',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])
```

What you don't see is:

* Fit/train (`model.fit()`)
* Evaluate with given metric (`model.evaluate()`)
* To add dropout after the `Convolution2D()` layer (or after the fully connected in any of these examples) a dropout function will be used, e.g., `Dropout(0.5)`
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

### PyTorch

Below is a ConvNet defined with the `Sequential` container in PyTorch ([Ref](https://pytorch.org/docs/master/nn.html?highlight=sequential#torch.nn.Sequential)).  This is a snippet with only the model definition parts - see the [References](#references) for the full code example.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetPyTorch(nn.Module):
    """Adapted from:
    https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/convnet.ipynb
    """
    def __init__(self, num_classes=10):
        super(ConvNetPyTorch, self).__init__()
        self.layer1 = nn.Sequential(
            # 28x28x1 => 28x28x32
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1), # (1(28-1) - 28 + 3) / 2 = 1
            nn.ReLU(),
            # 28x28x32 => 14x14x32
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2),
                         padding=0)) # (2(14-1) - 28 + 2) = 0    
        self.layer2 = nn.Sequential(
            # 14x14x32 => 14x14x64
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1), # (1(14-1) - 14 + 3) / 2 = 1   
            nn.ReLU(),
            # 14x14x64 => 7x7x64
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2),
                         padding=0)) # (2(7-1) - 14 + 2) = 0
        self.linear_1 = nn.Linear(7*7*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        logits = self.linear_1(out.view(-1, 7*7*64))
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

Below is a ConvNet defined with the `Layers` library and Estimators API in TensorFlow ([Ref](https://www.tensorflow.org/programmers_guide/estimators)).  This is a snippet with only the model definition parts - see the [References](#references) for the full code example.

```python
import tensorflow as tf

# Create the neural network
def convNetTensorFlow(x_dict, n_classes, reuse, is_training):
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

        # Output layer, class prediction
        logits = tf.layers.dense(fc1, n_classes, activation=None)
        
    return logits

"""...[snipped for brevity]"""

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

```

What you don't see is:

* Fit/train (`model.train()`)
* Evaluate with given metric (`model.evaluate()`)
* To add dropout after the `tf.layers.conv2d()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `tf.layers.dropout(inputs=net_layer, rate=0.5, training=is_training)`
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

For more see tensorflow in the [References](#references) below.

### Cognitive Toolkit (CNTK)

Below is a ConvNet defined with the `Layer` API in CNTK ([Ref](https://www.tensorflow.org/programmers_guide/estimators)).  This is a snippet with only the model definition parts - see the [References](#references) for the full code example (Note:  as of this writing CNTK is Windows or Linux only)

```python
import cntk as C

def convNetCNTK(features, num_output_classes):
    """https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html"""
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(2), lambda i: [
                C.layers.Convolution((3,3), [32,64][i], pad=True),
                C.layers.MaxPooling((2,2), strides=(2,2))
                ]),
            C.layers.Dense(64),
            C.layers.Dense(out_dims, activation=None)
        ])

    return model(features)

```

What you don't see is:

* Fit/train (`trainer = C.Trainer()` and `trainer.train_minibatch()`)
* Evaluate with given metric (`out = C.softmax()` and `out.eval()`)
* To add dropout after the `C.layers.Convolution()` layer (or even after the fully connected in any of these examples) a dropout function will be used, e.g. `C.layers.Dropout(0.5)`.
* Sometimes another fully connected (dense) layer with, say, ReLU activation, is added right before the final fully connected layer.

## Conclusion

No real conclusion except to say these frameworks do pretty much the same sorts of things and all have different API layers, high-level to low-level.

The full code samples are in this Jupyter [Notebook](https://github.com/michhar/python-jupyter-notebooks/blob/master/multi_framework/ConvNet_Comparisons.ipynb).

## References

Samples adapted in this post:

1.  Keras code sample [Ref](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
2.  PyTorch code sample [Ref](https://github.com/rasbt/deep-learning-book/blob/master/code/model_zoo/pytorch_ipynb/convnet.ipynb)
4.  TensorFlow code sample with Layers and Estimators APIs [Ref](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) and ConvNets Tutorial at this [Doc](https://www.tensorflow.org/tutorials/deep_cnn)
3.  CNTK code sample with Layer API [Doc](https://cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html)

A great book from which I took some of the concepts written in this post:  [Book](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition) and [Code]()

Even more nice code samples:

*  Kaggle Keras code sample:  https://www.kaggle.com/tonypoe/keras-cnn-example?scriptVersionId=589403
* Keras example:  http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
* PyTorch example: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py
* CNTK example:  https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html 
* TensorFlow Estimators example:  https://jhui.github.io/2017/03/14/TensorFlow-Estimator/

Thanks for reading.

## Appendix

Nice explanation of tensor layouts (PyTorch vs. TensorFlow) in a PyTorch forum post by Mamy Ratsimbazafy ([Post](https://discuss.pytorch.org/t/tensorflow-vs-pytorch-convnet-benchmark/8738/3):

> Furthermore there might be a difference due to the Tensor layouts:

> PyTorch use NCHW and Tensorflow uses NHWC, NCHW was the first layout supported by CuDNN but presents a big challenge for optimization (due to access patterns in convolutions, memory coalescing and such …).
NHWC is easier to optimize for convolutions but suffer in linear layers iirc because you have to physically transpose/permute the dimensions.

> Furthermore, due to it’s dynamic nature, PyTorch allocate new memory at each new batch while Tensorflow can just reuse previous memory locations since size is known in advance.

> Memory is THE bottleneck in Deep Learning not CPU, the big challenge is how to feed data fast enough to the CPU and GPU to get the maximum GFLOPS throughput.


<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
     */
    
    var disqus_config = function () {
        this.page.url = 'https://michhar.github.io/convolutional-in-layers-and-sequences/';  // Replace PAGE_URL with your page's canonical URL variable
        this.page.identifier = 'happycat1'; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };
    
    (function() {  // DON'T EDIT BELOW THIS LINE
        var d = document, s = d.createElement('script');
        
        s.src = 'https://michhar.disqus.com/embed.js';
        
        s.setAttribute('data-timestamp', +new Date());
        (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>