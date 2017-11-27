---
layout: post
title: "The Cognitive Toolkit (CNTK) Understands How You Feel"
img: yosemite_mharris.jpg
date: 2017-11-28 12:55:00 +0000
description: CNTK and tiny pictures of pills
tag: [cntk, deep-learning, computer-vision, machine-learning]
comments: true
---

![Circuits forming two profile pictures](http://www.archer-soft.com/en/sites/default/files/cntk.jpg)

**tl:dr**:  

## Some Background

### CNTK

The original name for Microsoft's CNTK was the _Computational Network Toolkit_, now known today simply as the _Cognitive Toolkit_, still abbreviated CNTK for short.  It was orignally written and offered up as a C++ package and now has Python bindings, making it much more widely adoptable.

> In its original words: [CNTK is] a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph

It was first open-sourced in April of 2015 with intended use for researchers and protoypers using GPUs for accelerated matrix calculations, much of what deep learning is built upon these days.  Interestingly, TensorFlow has its initial public release in November of 2015.  Of note, 2015 was also a good year for Microsoft Research in the computer vision space as they won the [ImageNet challenge](https://en.wikipedia.org/wiki/ImageNet#ImageNet_Challenge) that December using this toolkit and a 152-layer deep neural network.

Since the beginning CNTK has been available for Linux and Windows.  We will be using a Linux Docker image in a minute.

### Faces Data 

PICS data info goes here.

I wanted a good quality, verified data set so decided to sort through some existing emotion datasets (some labeled and some not).  I created folders first for 'sad' and 'happy' and just drag and dropped images from the datasets based on visual inspection in the file viewer.

> Tip:  creating folders to drag-n-drop files into made quick work of my sorting task and makes it easy to read in the "labels" or folder names later on programatically

Ideally, I'd get a couple other people to confirm my labels, but then again, this was just a proof of principle-type exercise so I didn't call in any favors this time.

## How I set things up

Since I'm on a Mac, I chose to use the Docker image of CNTK.  By following this [Doc](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Docker-Containers) I got a Jupyter notebook up and running with CNTK with all of the Tutorial notebooks at the ready and the ability to upload or create new ones as needed.

I ran these commands to get a Jupyter notebook set up with CNTK (v2.1 used here as it was more stable at the time of writing):

    docker pull microsoft/cntk:2.1-cpu-python3.5

    docker run -d --volume "$PWD/data:/data" -p 8888:8888 --name cntk-jupyter-notebooks -t microsoft/cntk:2.1-cpu-python3.5

    docker exec -it cntk-jupyter-notebooks bash -c "source /cntk/activate-cntk && jupyter-notebook --no-browser --port=8888 --ip=0.0.0.0 --notebook-dir=/cntk/Tutorials --allow-root"


I tried many different CNN network architectures (simple three-layer CNN, ones with pooling and dropout layers, etc.) and several hyperparameter combinations (minibatch sizes for training, learning rate, etc.).  I tried 64x64 pixel, 128x128 pixel, and 256x256 pixel 3-channel and 1-channel images.

## Positive Results

### Improvement #1:  Use sufficient data

I began this journey with about 1000 images that I sorted and curated into a "sad" or "happy" bucket - using the [PICS 2D face sets](http://pics.stir.ac.uk/2D_face_sets.htm).  I even got up to 72% accuracy with this dataset and a CNN with pooling.  Then I discovered a Kaggle competition from about 5 years ago dealing with recognizing facial expressions - [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and this provided me with about 13500 training images and 2300 test images of the "happy" or "sad" kind (there were also images for categories: angry, disgust, fear, surprise and neutral in case you wish to try these out).

### Improvement #2:  Intensity normalization

This improvement resulted in about a 5% increase in accuracy on the held-out test set.

### Improvement #3:  Add Pooling layers

Interestingly, layering in three pooling layers in between the convolutional layers (last one before the dense output layer) resulted in about another 5% jump in accuracy on the test set.

Training 28274 parameters in 8 parameter tensors.

### Improvement #4:  The learning rate

I took the learning rate down 10x (from 0.2 to 0.02) and my resulting accuracy increased approximately 5%.

There's much more I could have tried, but was on this day limited to a CPU only.  If you have GPU-acceleration options, give some more complex network architectures a try - you could spin up an Azure Deep Learning VM which is what I usually do at my day job.

## Conclusions

The improvements are not an exhaustive list of every and all combinations or permutations, but do represent some common ways to deal with image data using CNNs.  Happy deep learning!

## References

1.  [Historical (Jan. 2016) Readme on CNTK Repo](https://github.com/Microsoft/CNTK/tree/c977f3957d165ef2793ea9ee4d3f9165fc6c0b80)
2. [Microsoft researchers win ImageNet computer vision challenge](https://blogs.microsoft.com/ai/2015/12/10/microsoft-researchers-win-imagenet-computer-vision-challenge)
3.  [Facial Emotion Detection Using Convolutional Neural
Networks and Representational Autoencoder Units by Prudhvi Raj Dachapally](https://arxiv.org/pdf/1706.01509.pdf)


