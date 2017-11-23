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

The original name for Microsoft's CNTK was the _Computational Network Toolkit_, now known today simply as the _Cognitive Toolkit_, still abbreviated CNTK for short.  It was orignally written and offered up as a C++ package and now has Python bindings, making it much more widely adoptable.

> In its original words: [CNTK is] a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph

It was first open-sourced in April of 2015 with intended use for researchers and protoypers using GPUs for accelerated matrix calculations, much of what deep learning is built upon these days.  Interestingly, TensorFlow has its initial public release in November of 2015.  Of note, 2015 was also a good year for Microsoft Research in the computer vision space as they won the [ImageNet challenge](https://en.wikipedia.org/wiki/ImageNet#ImageNet_Challenge) that December using this toolkit and a 152-layer deep neural network.


## How I set things up

Since I'm on a Mac, I chose to use the Docker image of CNTK.  By following this [Doc](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Docker-Containers) I got a Jupyter notebook up and running with CNTK with all of the Tutorial notebooks at the ready and the ability to upload or create new ones as needed.

I ran these commands to get a Jupyter notebook set up with CNTK (v2.1 used here as it was more stable at the time of writing):

    docker pull microsoft/cntk:2.1-cpu-python3.5

    docker run -d --volume "$PWD/data:/data" -p 8888:8888 --name cntk-jupyter-notebooks -t microsoft/cntk:2.1-cpu-python3.5

    docker exec -it cntk-jupyter-notebooks bash -c "source /cntk/activate-cntk && jupyter-notebook --no-browser --port=8888 --ip=0.0.0.0 --notebook-dir=/cntk/Tutorials --allow-root"


## References

1.  [Historical (Jan. 2016) Readme on CNTK Repo](https://github.com/Microsoft/CNTK/tree/c977f3957d165ef2793ea9ee4d3f9165fc6c0b80)
1. [Microsoft researchers win ImageNet computer vision challenge](https://blogs.microsoft.com/ai/2015/12/10/microsoft-researchers-win-imagenet-computer-vision-challenge)


