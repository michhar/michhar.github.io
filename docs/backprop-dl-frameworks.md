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

Now, followed with these arrays which will hold the signals (remember these are gradients without their input terms mainly for convenience) (`oSignals` the one from output to hidden and `hSignals` the hidden to input layer).

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


## The Code

## Speed

## Conclusion

## References

1. 

Thanks for reading.