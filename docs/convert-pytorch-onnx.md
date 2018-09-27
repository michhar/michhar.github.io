---
img:  IMG_3568.JPG
layout: post
title: How to Convert a PyTorch Model to ONNX Format
comments: true
description: 
tags: [python, pytorch, onnx]
---

![](img/pytorch_loves_onnx.jpg)

Posted:  2018-09-27

It might seem tricky or intimidating to convert model formats, but ONNX makes it easier.  However we must get our PyTorch model into the ONNX format.  This involves both the weights and network architecture defined by a `nn.Module` inherited class.

I don't write out the model classes, however I wanted to share the steps and code from the point of having the class definition and some weights (either in memory or from a model path file).  One could also do this with the pre-trained models from the torchvision library.

## The General Steps

1. Define the model class if using a custom model
2. Train the model or load the weights (`.pth` file by convention) to something usually called the `state_dict`
3. Create a properly shaped input vector
5. (Optional) Give the input and output layers names (to later reference back)
6. Export to ONNX format with the PyTorch ONNX exporter

## Prerequisites

1. PyTorch and torchvision installed
2. A PyTorch model class and model weights

## Using a Custom Model Class and Weights File

The Python look something like:
```python
import torch
import torch.onnx

# A model class instance (class not shown)
model = MyModelClass()

# Load the weights from a file (.pth usually)
state_dict = torch.load(weights_path)

# Load the weights now into a model net architecture defined by our class
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(sample_batch_size, channel, height, width)

torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
```



The state dictionary, or `state_dict`, is a Python dict containing parameter values and persistent buffers.  ([Docs](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.load_state_dict))

> Note:  The preferred way of saving the weights is with `torch.save(the_model.state_dict(), <name_here.pth>)`. ([Docs](https://pytorch.org/docs/stable/notes/serialization.html#recommended-approach-for-saving-a-model))

## A Pre-Trained Model from torchvision

If using the `torchvision.models` pretrained vision models all you need to do is, e.g., for AlexNet:

```python
import torch
import torchvision.models as models

# Use an existing model from Torchvision, note it 
# will download this if not already on your computer (might take time)
model = models.alexnet(pretrained=True)

# Create some sample input in the shape this model expects
dummy_input = torch.randn(10, 3, 224, 224)

# It's optional to label the input and output layers
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

# Use the exporter from torch to convert to onnx 
# model (that has the weights and net arch)
torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
```

> Note, the pretrained model weights that comes with `torchvision.models` went into a home folder `~/.torch/models` in case you go looking for it later.

## Summary

Here, I showed how to take a pre-trained PyTorch model (a weights object and network class object) and convert it to ONNX format (that contains the weights and net structure).

## More References

1. [Example: End-to-end AlexNet from PyTorch to Caffe2](https://pytorch.org/docs/stable/onnx.html#module-torch.onnx)

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