---
img:  IMG_3568.JPG
layout: post
title: Back-Propagation in Deep Learning Frameworks
comments: true
description: A tech piece on back-propagation
cover:  /img/flask-post-diagram.png
tags: [python, deep-learning, tensorflow, cntk, keras, pytorch]
---

**tl;dr**:  

**Posted:**  2018-04-22

## Introduction

A rather lengthy introduction and code sample for back-propagation is written in NumPy below to set the stage for all future work in deep learning frameworks.  Then, how several deep learning frameworks perform/think about back-propagation follows.

I've found, recently, that the Sequential class in Keras and PyTorch are very similar to the Layer or Layers APIs in CNTK and TensorFlow - perhaps Sequential is a little bit higher-level so it depends on how much customizability you want, as usual in these cases.  Below you will see examples of the the same CNN architecture in the four different frameworks along with their back-propagation code.

### NumPy for Comparison

This will set the stage for working with deep learning frameworks such as TensorFlow and PyTorch.  NumPy is the currency of the data used in these frameworks.  It is good to have a solid grasp on backpropagation and for a deeper explanation see  [Wikipedia]().

To quote a [good article](https://blogs.msdn.microsoft.com/uk_faculty_connection/2017/07/04/how-to-implement-the-backpropagation-using-python-and-numpy/) that can say this better than me:

> "The goal of back-propagation training is to minimize the squared error. To do that, the gradient of the error function must be calculated. The gradient is a calculus derivative with a value like +1.23 or -0.33. The sign of the gradient tells you whether to increase or decrease the weights and biases in order to reduce error. The magnitude of the gradient is used, along with the learning rate, to determine how much to increase or decrease the weights and biases.  Using some very clever mathematics, you can compute the gradient."

The Neural Net class uses the following to kick-off the back propagation calculation (taken from this [Code](https://github.com/leestott/IrisData/blob/master/nn_backprop.py)).

Before anything a few variables were set up in the NeuralNetwork class.

```python
# Number of input, hidden and output nodes respectively
self.ni = numInput
self.nh = numHidden
self.no = numOutput

# The values on the nodes
self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

# The weight matrices
self.ihWeights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
self.hoWeights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)

# The bias matrices
self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)
```

Part of the training code to initialize the gradients and signals (like gradients without their input terms) and looks like the following:

```python
def train(self, trainData, maxEpochs, learnRate):
    ...

    hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output weights gradients
    obGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
    ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
    hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients
	
    oSignals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
    hSignals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms

    ...
```

> Pro tip:  "When working with neural networks, it's common, but not required, to work with the float32 rather than float64 data type" - Lee Stott

Now, followed with these arrays which will hold the signals(remember these are gradients without their input terms mainly for convenience) (`oSignals` the one from output to hidden and `hSignals` the hidden to input layer).

```python
oSignals = np.zeros(shape=[self.no], dtype=np.float32)
hSignals = np.zeros(shape=[self.nh], dtype=np.float32)
```

The calculation of the gradients, or amount used for the weight updates, are the steps as follows:

1. Compute output node signals (an intermediate value)
2. Compute hidden-to-output weight gradients using output signals
3. Compute output node bias gradients using output signals
4. Compute hidden node signals
5. Compute input-to-hidden weight gradients using hidden signals
6. Compute hidden node bias gradients using hidden signals

![Backprop calculation](/img/backprop/0617vsm_McCaffreyFig2s.jpg)

```python
# In setting up the training we had two variables
x_values = np.zeros(shape=[self.ni], dtype=np.float32)
# Expected values (training labels)
t_values = np.zeros(shape=[self.no], dtype=np.float32)

# 1. compute output node signals (an intermediate value)
for k in range(self.no):
    derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax
    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])  # E=(t-o)^2 do E'=(o-t)

# 2. compute hidden-to-output weight gradients using output signals
for j in range(self.nh):
    for k in range(self.no):
    hoGrads[j, k] = oSignals[k] * self.hNodes[j]
    
# 3. compute output node bias gradients using output signals
for k in range(self.no):
    obGrads[k] = oSignals[k] * 1.0  # 1.0 dummy input can be dropped
    
# 4. compute hidden node signals
for j in range(self.nh):
    sum = 0.0
    for k in range(self.no):
    sum += oSignals[k] * self.hoWeights[j,k]
    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
    hSignals[j] = derivative * sum
    
# 5 compute input-to-hidden weight gradients using hidden signals
for i in range(self.ni):
    for j in range(self.nh):
    ihGrads[i, j] = hSignals[j] * self.iNodes[i]

# 6. compute hidden node bias gradients using hidden signals
for j in range(self.nh):
    hbGrads[j] = hSignals[j] * 1.0  # 1.0 dummy input can be dropped

# update weights and biases using the gradients

...

```

Updating the weight matrix now with these gradients will be left up to the reader (or use that code sample link above).

### Keras

```python

def createCNNModel(num_classes):
    """ Adapted from: # http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
```

What you don't see is:



### PyTorch

(CNN)

```python
# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN()
```

### Tensorflow

CNN with the Layers and Estimators libraries

```python
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

[snipped]
```

For more see tensorflow in the [References](#references) below.

### Cognitive Toolkit (CNTK)

CNN for MNIST images with Layer API

```python
def create_basic_model(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        net = C.layers.Convolution((5,5), 32, pad=True)(input)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 32, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Convolution((5,5), 64, pad=True)(net)
        net = C.layers.MaxPooling((3,3), strides=(2,2))(net)

        net = C.layers.Dense(64)(net)
        net = C.layers.Dense(out_dims, activation=None)(net)

    return net
```

The following is the same CNN using the Layer API and, then, Sequential class to make the code compact.

```python
def create_basic_model_terse(input, out_dims):

    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
            C.layers.Dense(64),
            C.layers.Dense(out_dims, activation=None)
        ])

    return model(input)
```

    # Instantiate the trainer object to drive the model training
    lr = learning_parameter_schedule_per_sample(1)
    trainer = Trainer(z, (ce, pe), adadelta(z.parameters, lr), progress_writers)

    training_session(
        trainer=trainer,
        mb_source = reader_train,
        mb_size = minibatch_size,
        model_inputs_to_streams = input_map,
        max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
        progress_frequency=num_samples_per_sweep
    ).train()

    # Load test data
    path = os.path.normpath(os.path.join(data_dir, "Test-28x28_cntk_text.txt"))
    check_path(path)

    reader_test = create_reader(path, False, input_dim, num_output_classes)

    input_map = {
        feature  : reader_test.streams.features,
        label  : reader_test.streams.labels
    }

... [snipped]

```

### Maybe Caffe2

## The Code

## Speed

## Conclusion

## References

1.  Kaggle Keras code sample [Ref](https://www.kaggle.com/tonypoe/keras-cnn-example?scriptVersionId=589403)
2.  PyTorch code sample [Ref](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py)
3.  CNTK code sample with Layer API [Doc](https://cntk.ai/pythondocs/CNTK_201B_CIFAR-10_ImageHandsOn.html)
4.  TensorFlow code sample with Layers API [Doc](https://www.tensorflow.org/tutorials/layers)

Thanks for reading.