---
layout: post
title: Building PyTorch with LibTorch From Source with CUDA Support
comments: true
description: How to Build PyTorch Preview and Other Versions
cover:  
tags: [build, pytorch, pytorch-1.0, how-to]
---


![](img/pytorch_grows_up.jpg)


**tl;dr**:  Notes on building PyTorch 1.0 Preview and other versions from source including LibTorch, the PyTorch C++ API for fast inference with a strongly typed, compiled language.  So fast.

**Posted:**  2018-11-10

## Introduction

I'd like to share some notes on building PyTorch from source from various releases using commit ids.  This process allows you to build from any commit id, so you are not limited to a release number only.

I've used this to build PyTorch with LibTorch for Linux amd64 with an NVIDIA GPU and Linux aarch64 (e.g. NVIDIA Jetson TX2).

## Instructions

Create a shell script with the following contents (this being only an example) and refer to rest of post for possible changes you may have to make.

```bash
# Post 1.0rc1 for a few fixes I needed
PYTORCH_COMMIT_ID="8619230"

# Clone, checkout specific commit and build for GPU with CUDA support
git clone https://github.com/pytorch/pytorch.git &&\
    cd pytorch && git checkout ${PYTORCH_COMMIT_ID} && \
    git submodule update --init --recursive  &&\
    pip3 install pyyaml==3.13 &&\
    pip3 install -r requirements.txt &&\
    USE_OPENCV=1 \
    BUILD_TORCH=ON \
    CMAKE_PREFIX_PATH="/usr/bin/" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
    CUDA_BIN_PATH=/usr/local/cuda/bin \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
    CUDA_HOST_COMPILER=cc \
    USE_CUDA=1 \
    USE_NNPACK=1 \
    CC=cc \
    CXX=c++ \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py bdist_wheel

# Install the Python wheel (includes LibTorch)
pip3 install dist/*.whl

# Clean up resources
rm -fr pytorch
```

* Note, the size of the binary/wheel can be up to 180 MB.

### Build flag meanings

* USE_OPENCV=1 - build with OpenCV support
* BUILD_TORCH=ON - build LibTorch (C++ API)
* CMAKE_PREFIX_PATH="/usr/bin/" - where to find Python
* LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH - build lib paths
* CUDA_BIN_PATH=/usr/local/cuda/bin - where to find current CUDA
* CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ - where to find current CUDA Toolkit
* CUDNN_LIB_DIR=/usr/local/cuda/lib64 - where to find cuDNN install
* CUDA_HOST_COMPILER=cc - sets the host compiler to be used by nvcc
* USE_CUDA=1 - compile with CUDA support
* USE_NNPACK=1 - compile with cuDNN
* CC=cc - which C compiler to use for PyTorch build
* CXX=c++ - which C++ compiler to use for PyTorch build
* TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" - GPU architectures to accomodate
* TORCH_NVCC_FLAGS="-Xfatbin -compress-all" - extra `nvcc` (NVIDIA CUDA compiler driver) flags

### Changes to script that may be necessary

* Update `pip3` to `pip` as necessary (However, it's recommended to build with Python 3 system installs)
* Update `CMAKE_PREFIX_PATH` to your `bin` where Python lives
* Update `PYTORCH_COMMIT_ID` to one you wish to use.  Official release commit ids are
    * v0.3.1 - `2b47480` (which I still needed for a project)
    * v0.4.0 - `3749c58`
    * v0.4.1 - `a24163a`
    * v1.0rc1 - `ff608a9`
* If compiling on macOS, update to the following:
    * CC=clang
    * CXX=clang++
    * CUDA_HOST_COMPILER=clang
* To compile without CUDA support (e.g. on CPU-only), update to the following:
    * USE_CUDA=0
    * USE_NNPACK=0

## Resources

Two binaries are available here, built with:

- System: Linux Ubuntu 16.04
- Machine: x86_64
- CUDA:  9.2
- GPU:  NVIDIA GTX 1080

Binaries built with above system:

| PyTorch Version or Commit ID | Download Link |
| --- | --- |
| 1.0 (commit id: 8619230) 94 MB | https://generalstore123.blob.core.windows.net/pytorchwheels/torch-1.0.0a0+8619230-cp35-cp35m-linux_x86_64.whl |
| 0.3.1 (commit id:  2b47480) 172 MB | https://generalstore123.blob.core.windows.net/pytorchwheels/torch-0.3.1b0+2b47480-cp35-cp35m-linux_x86_64.whl |

* aarch64 binaries coming soon.

## Conclusion

Given the right hardware (Linux amd64 or even aarch64 like a TX2) - the above script will work to build PyTorch and LibTorch.  Leave a comment if you wish - issues or suggestions welcome.

## References

1.  [PyTorch official Dockerfile](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
2.  [Micheleen's GPU VM Dockerfile with a PyTorch+LibTorch build included](https://github.com/michhar/custom-jupyterhub-linux-vm/blob/master/Linux_py35_GPU.dockerfile#L199)
3.  [NVCC from NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

## Thank Yous

To PyTorch GitHub Issues with great activity and insights (https://github.com/pytorch/pytorch/issues) and the official PyTorch Forums (https://discuss.pytorch.org/).


<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
     */
    
    var disqus_config = function () {
        this.page.url = 'https://michhar.github.io/how-i-built-pytorch-gpu/';  // Replace PAGE_URL with your page's canonical URL variable
        this.page.identifier = 'happycat2'; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };
    
    (function() {  // DON'T EDIT BELOW THIS LINE
        var d = document, s = d.createElement('script');
        
        s.src = 'https://michhar.disqus.com/embed.js';
        
        s.setAttribute('data-timestamp', +new Date());
        (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>