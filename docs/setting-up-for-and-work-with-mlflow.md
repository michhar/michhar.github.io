---
img:  IMG_3568.JPG
layout: post
title: Working with MLFlow - A Reproducible Project Pattern for ML Training and Deployments
comments: true
description: Sample using a new OSS tool, MLFlow, for an entire Keras Object Detection workflow
cover:  /img/flask-post-diagram.png
tags: [python, keras, yolov3, mlflow]
---

![header pic]()

**tl;dr**:  

**Posted:**  2019-09-15

## Introduction to MLFlow

I discovered MLFlow on a hunt for a way to track my models and environment, parameters, training and deployments (that was a nice addition) in a single project and a single place.

A few features that caught my eye:

* Run training and deployment from a remote GitHub repo
* Log the environment in which a model was trained (other logging and even a UI for tracking)
* Many deployment options (local, Azure, AWS, etc.)

## The Sample Project

Feel free to follow along or create your own project.  This is an example of using MLFlow with an existing repo.

1.  Clone the YOLOv3 Keras repo here:  

> `git clone https://github.com/michhar/keras-yolo3.git keras-yolo3-mlflow`

2.  CD into model directory:

> `cd keras-yolo3-mlflow/model_data/`

3.  Download the model, data (or create your own) and pointer file to data.
  * Get a Keras-friendly YOLOv3 base model, converted directly from the Tiny YOLOv3 Darknet model (here, the Tiny weights are used - nice for constrained devices):

    > Click to access through Google Drive [here (34MB)](https://drive.google.com/open?id=1VulLSbrFrshPkEy71RgNhoVPpHdxEyrl) and download

  * Get some sample data of lego minifigures with helmets and other head gear to train a model to detect what the figurine is wearing on its head, placing the uznipped folder in the `voc` subdirectory.

    > Click to access through Google Drive [here (173MB)](https://drive.google.com/open?id=1NbzE9rVPoBsFNZvJ7zso4GLsTW6vjnZW) or use your own data (according to instructions on the repo [in Step 1]((https://github.com/michhar/keras-yolo3#training) - you'll need to label as well).

  * Get the list of images as a small text file with associated bounding boxes and class.

    > Click to access through Google Drive [here](https://drive.google.com/open?id=1mbtjG0s1dMOsiWd-clCV2LXdGKJ02Yqa)


## Tracking

## Training

## Deploying with Azure Machine Learning

## Conclusion

<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
     */
    
    var disqus_config = function () {
        this.page.url = 'https://michhar.github.io/setting-up-for-and-work-with-mlflow/';  // Replace PAGE_URL with your page's canonical URL variable
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