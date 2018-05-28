---
layout: post
title: Building and Running ELL on MacOS for Object Detection Part 1
comments: true
description: How to Build ELL on macOS
cover:  
tags: [object-detection, how-to]
---

# Building ELL for macOS

Out of Microsoft Research and in early preview, is the Embedded Learning Library, a toolkit for deploying pre-made, small models to a resource contrained system, like a Raspberry Pi 3.  The models are designed to run without any connection to the cloud resulting in a truly disconnected device use case.

## Walkthrough

My macOS specs:

![mac specs](img/mac-specs-20180528.png)

As the writing of this article, release 2.3.0 of the Embedded Learning Library (ELL) was successfully built.

On macOS, to build, follow this tutorial:  https://github.com/Microsoft/ELL/blob/master/INSTALL-Mac.md, except for the `llvm` install (see below) and ensure that you are using the full path to llvm when using `cmake`, as in:

    cmake -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm ..

The following is based on reference (1) (see [References](#references)).

Get llvm 3.9.x (need 3.9 to work with ELL):

    brew install --with-toolchain llvm

Get a list of local version and the active version:

    brew search llvm

Optional (in case you don't get llvm 3.9.x):

    brew update
    brew upgrade

Find the binaries:

    (brew --prefix llvm)/bin

Added to `~/.bashrc` file:

```bash
# For building ELL
export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include -I/usr/local/opt/llvm/include/c++/v1/"
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
```

    source ~/.bashrc

To build the python bindings ensure you run `make` with the right flag:  `make _ELL_python`

## My Troubleshooting

* Getting the right C and C++ compiler specified - these must be the llvm versions for ELL to build.
* If something goes awry and the build continuously fails, the last resort is to delete the whole `build` folder and then recreate it and follow the rest of the instructions.
* Make sure using the brew installed packages.  These should be the ones in the `/usr/local/opt`.
* Follow tutorial instructions for using the right LLVM module at their [troubleshooting](https://github.com/Microsoft/ELL/blob/master/INSTALL-Mac.md#troubleshooting)

## References

1. https://embeddedartistry.com/blog/2017/2/20/installing-clangllvm-on-osx


