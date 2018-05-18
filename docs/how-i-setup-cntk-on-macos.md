---
layout: post
title: Building CNTK on MacOS
comments: true
description: How to Build CNTK on macOS
cover:  
tags: [cntk, deep-learning, how-to]
---

**tl;dr**:  Notes on building CNTK (Cognitive Toolkit) on macOS.

**Posted:**  2018-05-16

Following:  https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux with some modifications.

Ensure XCode is installed

Proceed with steps in article with the following modifications.

## My System (by the way)

<img src=img/my_sys.png width=50%>

`g++ --version`:

```
    Apple LLVM version 9.0.0 (clang-900.0.39.2)
    Target: x86_64-apple-darwin17.3.0
    Thread model: posix
    InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

## MKL

For https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#mkl (MKL install) use the following:

Create the directory for MKL:  `sudo mkdir /usr/local/mklmil`

Create a file `build_mklml.sh` and add the following to it.

```bash
sudo wget https://github.com/intel/mkl-dnn/releases/download/v0.14/mklml_mac_2018.0.3.20180406.tgz && \
sudo tar -xzf mklml_mac_2018.0.3.20180406.tgz -C /usr/local/mklml && \
wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.12.tar.gz | tar -xzf - && \
cd mkl-dnn-0.12 && \
ln -s /usr/local external && \
mkdir -p build && \
cd build && \
cmake .. && \
make && \
make install && \
cd ../.. && \
rm -rf mkl-dnn-0.12
```

Modify `build_mklml.sh` to be executible (`chmod +x build_mklml.sh`) and run (`./build_mklml.sh`).

## OpenMPI

This is not a modification, just a code snippet for a bash script.

```bash
wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz && \
tar -xzvf ./openmpi-1.10.3.tar.gz && \
cd openmpi-1.10.3 && \
./configure --prefix=/usr/local/mpi && \
make -j all && \
sudo make install
```

## Protobuf

Skipped the `apt-get` commands.

This is not a modification, just a code snippet for a bash script.

```bash
wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz && \
tar -xzf v3.1.0.tar.gz && \
cd protobuf-3.1.0 && \
./autogen.sh && \
./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/protobuf-3.1.0 && \
make -j $(nproc) && \
sudo make install
```

## ZLIB

I did not need to `apt-get` or download/install this library as it was already installed, but if not on your system and the correct version, follow the instructions in the article.

## LIBZIP

This is not a modification, just a code snippet for a bash script.

```bash
wget http://nih.at/libzip/libzip-1.1.2.tar.gz && \
tar -xzvf ./libzip-1.1.2.tar.gz && \
cd libzip-1.1.2 && \
./configure && \
make -j all && \
sudo make install
```

## Boost

Skipped the `apt-get` commands.

Ran as a bash script:

```bash

```

It can take some time.