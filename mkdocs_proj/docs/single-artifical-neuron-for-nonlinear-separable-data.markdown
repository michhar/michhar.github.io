---
layout: post
title: "On using an Adaline Artificial Neuron for Classification"
img: chimi_mharris.jpg
date: 2017-07-19 12:55:00 +0800
description: Some fun with activation functions and dimensionality reduction
tag: [single-layer, artificial-neuron, adaline, PCA, leukemia, sigmoid-activation]
comments: true
---

**tl:dr**:  Getting a simple, predictive framework distinguishing two types of leukemia based on biological markers from a single-layer neural network was not the intent of this exercise. It is, however, indicative of the power of a single artificial neuron and thoughtful feature reduction.

**Posted:**  2017-07-19

### Introduction

The intent of this post originally was to show the inner workings and limitations of a single artificial neuron using some moderately complex, noisy data; a challenge of sorts - "is this noisy data linearly separable with a single artificial neuron and if not, why is that?".  

However, I found with some data and algorithm exploration, that I could distinguish between two types of leukemia — a naive approach and not really biologically significant, but an interesting outcome nonetheless.  So, even though this post is about the data science, it also touches on a potential method to use in the real world.

In this post, you'll find information on the use of PCA for data reduction/feature engineering, scaling and normalization for preprocessing, the Adaline algorithm (artificial neuron), different activation functions, among other topics and concepts.

- [What is an Adaline artificial neuron](#what-is-an-adaline-artificial-neuron)
- [Adaline with a sigmoid activation function](#adaline-with-a-sigmoid-activation-function)
- [Choosing an activation function](#choosing-an-activation-function)
- [The noisy data](#the-noisy-data)
- [3D to run through network and 2D to gain insights](#3d-to-run-through-network-and-2d-to-gain-insights)
- [Conclusion from my experiment](#conclusion-from-my-experiment)
- [Credits and further reading](#credits-and-further-reading)


### What is an Adaline artificial neuron

The ADAptive LInear NEuron (Adaline) algorithm is very similar to a Perceptron (simplest of the artificial neurons) except that in the Perceptron the weights are updated based on a unit step activation function output (see figure below) whereas Adaline uses a linear activation function to update it's weights giving it a more robust result (that even converges with samples that are not completely separable by a linear hyperplane, unlike the Perceptron).  In Adaline a _quantizer_ after the activation function, is used to then predict class labels.

Beyond the linear activation function and the _quantizer_, we see the use of a _cost function_, or _objective function_, to update the weights.  In this case we want to minimize this function with an optimization method.  The optimization of the _cost function_ happens with yet another function aptly and simply named an _optimization function_.  In this case our optimization function is _stochastic gradient decent_, which one can of as "climbing down a hill" (using part of the data to calculate, shuffled as well) to get to the minima of the cost function's convex curve (as it updates weights iteratively from a shuffled dataset).

A really great discussion from which much of this information was adapted can be found in Sebastian Raschka's _Python Machine Learning_ book (link [here](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning)) and excellent blog post on this topic [here](http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html) on the single-layer neurons.

### Adaline with a sigmoid activation function

I grabbed Raschka's ADAptive LInear NEuron (Adaline) classifier open-source code [here](https://github.com/PacktPublishing/Python-Machine-Learning/blob/master/3547_02_Code.ipynb) (the AdalineSGD class) and updated the activation function to logistic sigmoid from a linear function.

Note, with the Adaline (versus the Perceptron) we use a continuous number rather than the binary class label, to compute the model error and update the weights.  Then to predict a class label, another function is used called a _quantizer_.  Also, the weights are updated in a more sophisticated manner.


### Choosing an activation function

![](/img/single_layer_neuron/singleneuron_activation.png)

In code, given this "net input" function:

```python
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
```

I update the activation function from linear as in:   

```python
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
```

To a logistic sigmoidal function:

```python
    def activation(self, X):
        """Compute sigmoidal activation
        
        Returns
        -------
        A 1d array of length n_samples

        """
        x = self.net_input(X)
        func = lambda v: 1 / (1 + math.exp(-v))
        return np.array(list(map(func, x)))

```

Full code [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/leukemia_notebook.ipynb) and [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/adaline_sgd.py).

**We still get linear classification boundaries**

These single-neuron classifiers can only result in linear decision boundaries, even if using a non-linear activation, because it's still using a single threshold value, `z` as in diagram above, to decide whether a data point is classified as 1 or -1.

### The noisy data

The data was downloaded from the Machine Learning Data Set Repository [mldata.org](https://mldata.org) using a convenience function from `scikit-learn`.  

```python
from sklearn.datasets.mldata import fetch_mldata

# Fetch a small leukemia dataset from mldata.org
#   http://mldata.org/repository/data/viewslug/leukemia-all-vs-aml/
test_data_home = tempfile.mkdtemp()
leuk = fetch_mldata('leukemia', transpose_data=True,
                      data_home=test_data_home)
```

The data is a small, but wide acute lymphocytic leukemia (ALL) vs. acute myelogenous leukemia (AML) dataset.  It has approximately 7000 biological markers (our features), vs. 72 samples (our data points).

Given the noisy nature of the data and possible skewedness, it was standardized and normalized with convenience functions from `scikit-learn`:

```python
from sklearn.preprocessing import RobustScaler, Normalizer

# Fit the scalar to the training dataset for 
#   zero mean and unit variance of features.
#   Using a robust scaler which is more resistent to outliers.
scaler = RobustScaler()
scaler.fit(X_train)

# Apply the transform
X_train = scaler.transform(X_train)

# Apply the same transform to the test dataset 
#   (simulating what happens when we get new data)
X_test = scaler.transform(X_test)

# Normalizing data as well to scale samples to unit norm
normalizer = Normalizer().fit(X_train)
X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)
```

Full code [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/leukemia_notebook.ipynb) and [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/adaline_sgd.py).

I tried just one feature reduction with PCA to reduce all 7129 dimensions to 2D at first.  However, I could not separate out the ALL samples from AML - this wasn't necessarily important to my post on Adaline neurons I was writing, but I decided to try something I'd read about recently for kicks.  In fact the idea sprung from a comment in a Python script where a perceptron was used to create non-linear separation of data for a plot (from [this](https://github.com/daniel-e/pymltools/blob/master/plot_scripts/plot_perceptron_nonlin.py) script on Github).  The comment went:

```
# map the data into a space with one addition dimension so that
# it becomes linearly separable
```

So, I gave it a shot.

### 3D to run through network and 2D to gain insights

My next step was to try feeding the neural network the data in 3D space (the 3 features or components from the first PCA reduction).

I then reduced the 3D data to 2D, mainly to visualize it.  A hyperplane was drawn (blank dashed line) to represent the decision boundary.  The surface in the diagram below is representative of a sigmoidal output along the direction of the weight vector.

![](/img/single_layer_neuron/linearly_sep_leukemia.png)

Full code [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/leukemia_notebook.ipynb) and [here](https://github.com/michhar/python-jupyter-notebooks/blob/master/machine_learning/adaline_sgd.py).

Note, the stochastic part of the single-neuron optimizer, stochastic gradient decent, causes some variation in the results if run again.  It might be a good idea to do a batch version of the Adaline neuron.  Another note is that one does not necessarily have to use a logistic sigmoidal activation function; it was just used here as an experiment and to prove to myself I'd always get a linear decision boundary.

### Conclusion from my experiment

I was surprised and impressed that I got a linearly separable result!  Albeit, that was not the intent of this exercise, but indicative of the power of a single neuron and thoughtful feature reduction.  It makes me wonder what a small neural network could do!


### Credits and further reading

1. Sebastian Raschka's _Python Machine Learning_ [book](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning)
2. The open-source notebooks with code accompanying the _Python Machine Learning_ book [here](https://github.com/PacktPublishing/Python-Machine-Learning) and related code [here](https://github.com/rasbt/mlxtend/tree/master/mlxtend/classifier)
2. Raschka's blog [post](http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html) on _Single-Layer Neural Networks and Gradient Descent_
3. `Scikit-learn`'s preprocessing data module [link](http://scikit-learn.org/stable/modules/preprocessing.html) for scaling features and samples
