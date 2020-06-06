---
layout: post
title: "Lessons from YOLO v3 Implementations in PyTorch"
date: 2019-11-23 12:55:00 +0000
description: Lessons from working on a YOLO v3 training script and custom data loader in PyTorch
tags: [pytorch, yolov3]
comments: true
---

# Lessons from YOLO v3 Implementations in PyTorch

![yolov3](img/pl14_out.jpg)
<br>

**UPDATE 2020-06-06**:  YOLO v4 has been recently published and implemented in PyTorch and can be found at https://github.com/Tianxiaomo/pytorch-YOLOv4.

**tl:dr**:  YOLO (for "you only look once") v3 is a relatively recent (April 2018) architecture design for object detection.  PyTorch (recently merged with Caffe2 and production as of November 2018) is a very popular deep learning library with Python and C++ bindings for both training and inference that is differentiated from Tensorflow by having a dynamic graph.  This post is about my lessons working on PyTorch YOLO v3 and a little insight into creating a good YOLO v3 custom model on custom data (_We love you COCO, but we have our own interets, now._).

**Posted:**  2019-11-23

## Quick Links

- <a href="https://arxiv.org/pdf/1804.02767.pdf" target="_blank">Original YOLO v3 paper</a>
- <a href="https://github.com/ayooshkathuria/pytorch-yolo-v3" target="_blank">Original PyTorch codebase</a>
- <a href="https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/" target="_blank">Ayoosh Kathuria's original blog post on implementing YOLO v3 in PyTorch</a>

## Lessons

### Anchor boxes (and briefly how YOLO works)

In order to understand the anchors or anchor boxes, a little background is needed on the YOLO v3 algorithm (sources are the original YOLO and YOLO v3 papers).

In full-sized YOLO v3 there are 9 anchor boxes specified in total as can be seen in the <a href="https://github.com/michhar/pytorch-yolo-v3-custom/tree/master/cfg" target="_blank">`cfg` files</a> on the PyTorch repo. 

    [[94, 89], [188, 190], [322, 308], [401, 401], [483, 475], [555, 539], [634, 646], [771, 765], [960, 866]]

There are 3 scales at which YOLO "sees" an image when passes through the network (these correspond to the three `yolo` layers).  Note, this allows YOLO to see big, medium and small sized objects all at once.

At each of the three scales, the image is broken in to a grid of 13x13 squares or cells (remember, our input image is converted to a `416x416` square in this implementation before running through the network).  For each cell in a 13x13 grid, three anchor boxes are used (this corresponds to the three anchor boxes from above).  In other words, each cell has three anchor boxes overlayed on it and this happens at three different scales (all within the same pass through the network, even!  Hence, "you only look once" :-) ).

So, when we list the array of 9 anchor boxes from above, the first three width/heights (`[94, 89], [188, 190], [322, 308]`), belong to the first scaling process, the second three (`[401, 401], [483, 475], [555, 539]`) to the second scaling process and, as follows, the final three to the third scaling process (`[634, 646], [771, 765], [960, 866]`).  Each set of three width/heights correspond to the width/heights of the three bounding boxes used for each grid cell at each of the three scales.

To round out this story, the three anchor boxes are used to predict whether there is an object there (object/no object).  The grid cell is used to predict classes.  These are combined at the end of the network to figure out the shape of objects (bounding boxes) from anchor boxes and their classes from grid cells.

This diagram shows this very well (anchor boxes on top path and grid cell predictions on bottom path):

![](https://imgs.developpaper.com/imgs/1055126480-5cde54b80a7c3_articlex.png)

<div align="right"><a href="https://pjreddie.com/media/files/papers/yolo.pdf" target="_blank">Image source</a></div>

With this all being said, the lesson is to always calculate the anchor boxes on each new dataset before training.  The sizes of labeled objects (which determines sizes of anchor boxes) will be crucial to a good training experiment and well as inference which uses the same anchor box sizes.

Anchor boxes are calculated using Kmeans clustering for every new dataset as is shown in code <a href="https://github.com/michhar/pytorch-yolo-v3-custom/blob/master/scripts/kmeans.py" target="_blank">here</a> (adapted from a Keras implementation of YOLO v3).

### Transfer learning

In transfer learning we begin with a base model which gives us the weight values to start our training.  Objects from the training set of the base model, upon which the base model was trained, gets us closer to a new learned network for objects in the real world.  So, instead of starting with random weights to begin our training we begin from a "smarter" set of values.

- One tidbit I learned was to skip making batch normalization (BN) layers trainable.

I recently learned from <a href="https://medium.com/luminovo/a-refresher-on-batch-re-normalization-5e0a1e902960" target="_blank">A refresher on batch (re-)normalization</a> that:

_"When the mini-batch mean (µB) and mini-batch standard deviation (σB) diverge from the mean and standard deviation over the entire training set too often, BatchNorm breaks."_

And that there are perils in hyperparameter tuning in conjunction with retraining BN layers and a few extra steps required to fix this (with a technique call batch renormalization) - so for simplicity sake, I left out retraining on BN layers, but look at batch renormalization techniques in the post above for addressing the complex issue if you wish.

How to allow layers in a PyTorch model to be trainable (minus BNs).

<pre>
<code class="language-python">
# Freeze layers according to user specification
stop_layer = layers_length - args.unfreeze # Freeze up until this layer
cntr = 0

for name, param in model.named_parameters():
    if cntr < stop_layer:
        param.requires_grad = False
    else:
        if 'batch_norm' not in name:
            print("Parameter has gradients tracked.")
            param.requires_grad = True
        else:
            param.requires_grad = False
    cntr+=1
    </code>
</pre>

### Finetuning

- How much of network to "open up" or set as trainable (the parameters that is)? - it's recommended at times to open it more (likely all of the parameters in fine-tuning phase) if the object or objects are very different from any COCO classes, which is called domain adaptation (NB:  the `yolov3.weights` base model from darknet is trained on COCO dataset).  So, for instance, if the base model has never seen a caterpillar before (not in COCO), you may want to let more layers be trainable.

How to allow even more layers in the PyTorch model to be trainable (could set `stop_layer` to 0 to train whole network):

<pre>
<code class="language-python">
# "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked - backprop)
stop_layer = layers_length - (args.unfreeze * 2) # Freeze up to this layer (open up more than first phase)

"""...[same as above section]"""
    </code>
</pre>

- Another learning is that if the network is not converging, try opening up all of the layers during fine-tuning.

### Data augmentation

Some of these I learned the hard way, others from the wonderful PyTorch forums and StackOverflow.

- Be careful of conversions from a 0-255 to a 0-1 range as you don't want to do that more than once in code.
- Keep this simple at first with only the resize and normalization.  Try with several types of augmentation next, increasing in complexity with each experiment.

Start with just resize and standard pixel intensity normalize.  (NB:  the transforms operate on PIL images, then convert to `numpy` 3D array and finally to `torch.tensor()`)

<pre>
<code class="language-python">
custom_transforms = Sequence([YoloResizeTransform(inp_dim), Normalize()])
    </code>
</pre>

Then get fancier with hue, saturation and brightness shifts, for example (look in `cfg` for the amounts if following along in <a href="https://github.com/michhar/pytorch-yolo-v3-custom" target="_blank">code</a>).

<pre>
<code class="language-python">
custom_transforms = Sequence([RandomHSV(hue=hue, saturation=saturation, brightness=exposure), 
    YoloResizeTransform(inp_dim), Normalize()])
    </code>
</pre>

Where Normalize is a pixel intensity normalization (here, not to unit norm because we do that elsewhere) (based on <a href="https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues" target="_blank">accepted answer on StackOverflow</a>):

<pre>
<code class="language-python">
class Normalize(object):
    """Pixel-intensity normalize the input numpy image"""

    def __init__(self):
        self.channels = 3

    def __call__(self, img, bboxes):
        """
        Args:
            img : numpy array
                Image to be scaled.
        Returns:
            img : numpy array
                normalize image.
        """
        arr = img.astype('float')
        # Do not touch the alpha channel
        for i in range(self.channels):
            minval = arr[...,i].min()
            maxval = arr[...,i].max()
            if minval != maxval:
                arr[...,i] -= minval
                # Don't divide by 255 because already doing elsewhere
                arr[...,i] *= ((maxval-minval))
        return arr, bboxes
    </code>
</pre>

- A great option for augmentation is to double or triple the size of a dataset with a library like `imgaug` which can handle bounding boxes and polygons now.

### Learning rate schedulers

There are some great learning rate schedulers to decrease learning rate with training on a schedule or automatically in the <a href="https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate">`torch.optim.lr_scheduler`</a> and set of methods therein.

The following is more of an implementation detail, but nonetheless, found it helpful to not make the mistake.

- Place the learning rate scheduler at the level of the epoch update, **not** the inner loop over batches of data (where the optimizer is).


## YOLO Glossary

- YOLOv3:  You Only Look Once v3.  Improvments over v1, v2 and YOLO9000 which include [Ref](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b):
    - Predicts more bounding boxes per image (hence a bit slower than previous YOLO architectures)
    - Detections at 3 scales
    - Addressed issue of detecting small objects
    - New loss function (cross-entropy replaces squared error terms)
    - Can perform multi-label classification (no more mutually exclusive labels)
    - Performance on par with other architectures (a bit faster than SSD, even, in many cases)
- Tiny-YOLOv3:  A reduced network architecture for smaller models designed for mobile, IoT and edge device scenarios
- Anchors:  There are 5 anchors per box.  The anchor boxes are designed for a specific dataset using K-means clustering, i.e., a custom dataset must use K-means clustering to generate anchor boxes.  It does not assume the aspect ratios or shapes of the boxes. [Ref](https://medium.com/@vivek.yadav/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807)
- Loss:  using `nn.MSELoss` (for loss confidence) or mean squared error
- IOU:  intersection over union between predicted bounding boxes and ground truth boxes

## References

1.  <a href="https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607" target="_blank">37 Reasons why your Neural Network is not working</a>
2. <a href="https://github.com/aleju/imgaug" target="_blank">`imgaug` augmentation Python library</a>
3.  <a href="https://machinethink.net/blog/object-detection-with-yolo/" target="_blank">Real-time object detection with YOLO</a>
4.  <a href="https://medium.com/luminovo/a-refresher-on-batch-re-normalization-5e0a1e902960" target="_blank">A refresher on batch (re-)normalization</a>


<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
     */
    
    var disqus_config = function () {
        this.page.url = 'https://michhar.github.io/learning-from-learning-yolov3/';  // Replace PAGE_URL with your page's canonical URL variable
        this.page.identifier = 'happycat3'; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };
    
    (function() {  // DON'T EDIT BELOW THIS LINE
        var d = document, s = d.createElement('script');
        
        s.src = 'https://michhar.disqus.com/embed.js';
        
        s.setAttribute('data-timestamp', +new Date());
        (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>