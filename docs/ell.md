---
layout: post
title: Building and Running ELL on MacOS for Object Detection
comments: true
description: How to Build and Run ELL on macOS
cover:  
tags: [cntk, deep-learning, how-to]
---

## Building ELL

On mac, to build, follow this tutorial:  https://github.com/Microsoft/ELL/blob/master/INSTALL-Mac.md

For locally installed versions:
    
    `brew list llvm --versions`

* Make sure using the brew installed packages.  These should be the ones in the `/usr/local/opt`.
* Follow tutorial instructions for using the right LLVM module at their [troubleshooting](https://github.com/Microsoft/ELL/blob/master/INSTALL-Mac.md#troubleshooting)