---
img:  IMG_3568.JPG
layout: post
title: Working with MLflow - A Reproducible Project Pattern for ML Training and Deployments
comments: true
description: Sample using a new OSS tool, MfFlow, for an entire Keras Object Detection workflow
cover:  /img/flask-post-diagram.png
tags: [python, keras, yolov3, mlflow]
---

![header pic]()

**tl;dr**:  

**Posted:**  2019-09-15

## Introduction to MLFlow

Recently, I discovered MLflow on a hunt for a way to track my models and environment, parameters, training and deployments (that was a nice addition) in a single project and a single place.

A few features that caught my eye:

* Run training and deployment from a remote GitHub repo
* Log the environment in which a model was trained (other logging and even a UI for tracking)
* Many deployment options (local, Azure ML, AWS, etc.)

After trying MLflow, I discovered that I liked:

* Error messaging is clear and includes a stacktrace


## Get Model and Data

Feel free to follow along or create your own project.  This is an example of using MLFlow with an existing repo.

1.  Clone the YOLOv3 Keras example repo:  

> `git clone https://github.com/michhar/mlflow-keras-example.git`

2.  cd into model directory:

> `cd mlflow-keras-example/model_data/`

3.  Download the model, data (or create your own) and pointer file to data.
    * Get a Keras-friendly YOLOv3 base model, converted directly from the Tiny YOLOv3 Darknet model (here, the Tiny weights are used - nice for constrained devices):

    > Click to access through Google Drive [here (34MB)](https://drive.google.com/open?id=1VulLSbrFrshPkEy71RgNhoVPpHdxEyrl), download and place model in `model_data` subdirectory

    * Get some sample data of lego minifigures with helmets and other head gear to train a model to detect what the figurine is wearing on its head, placing the uznipped folder in the `voc` subdirectory.

    > Click to access through Google Drive [here (173MB)](https://drive.google.com/open?id=1NbzE9rVPoBsFNZvJ7zso4GLsTW6vjnZW) or use your own data (according to instructions on the repo [in Step 1]((https://github.com/michhar/keras-yolo3#training) - you'll need to label as well).

    * Get the list of images as a small text file with associated bounding boxes and class.

    > Click to access through Google Drive [here](https://drive.google.com/open?id=1mbtjG0s1dMOsiWd-clCV2LXdGKJ02Yqa) and place it in the `voc` subdirectory

## Setup for MLflow

Required Python packages:
  * `mlflow`
  * `azure-cli`

Also run, for the AzureML deployment CLI:

    `pip install -r https://aka.ms/az-ml-o16n-cli-requirements-file`

`MLproject` file is an excellent source of control over things.  Optional, but recommended for the following reasons:
  * Points to conda dependencies file for building this environment before training
  * Parameters, types and defaults
  * Entrypoint command (with all options) - this is the master command that is run when the mlflow training is run on the command line

A simplified `MLproject` file is as follows:

```
name: My Awesome Project

conda_env: smart_conda.yaml

entry_points:
  main:
    parameters:
      parameter1: {type: str, default: "voc/list_master.txt"}
      parameter2: {type: float, default: 1e-2}
      parameter3: {type: int, default: 16}
    command: "python train.py --parameter1 {parameter1} --parameter2 {parameter2} --parameter3 {parameter3}"
```

If there are defaults, none of these parameters need to exist on the command line when running with `mlflow`, however they may be overridden.

## Training

To train, all you should need to do from within the cloned repo folder is (runs with default parameters in MLproject entrypoint command):

    `mlfow run .`

Or if you want to modify a default parameter or two (use `-P` per parameter) like the number of epochs for the transfer learning stage (`frozen_epochs`) and network fine tuning stage (`fine_tune_epochs`):

    `mlflow run . -P frozen_epochs=5 -P fine_tune_epochs=3`

Also, you can monitor the run through `tensorboard` which is part of a callback in the `model.fit` method (change logdir as appropriate).

    `tensorboard --logdir logs/default`

The metric and model (as an artifact) is recorded by the following:

    ```python
    # Added for MLflow
    mlflow.keras.log_model(model, "keras-yolo-model-frozen-pass")
    mlflow.log_metric('frozen_loss', history.history['val_loss'][-1])
    ```

What does `mlflow run` actually do?  As follows:

  * Creates the conda environment based on the `yolo_conda.yml` file (or whatever you named the dependencies file)
  * Runs the entrypoint script described in the `MLproject` file (this file is not mandatory, but makes customizing the training process much easier).
  * Logs any artifacts specified in `train.py` (e.g. save model as an asset with `mlflow.keras.log_model(model, "keras-yolo-model-frozen-pass")`)
  * Logs any metrics specified in `train.py` (e.g. log loss with `mlflow.log_metric('finetune_loss', model.loss)`)


### Artifacts

If model is logged as in this training script, a folder should appear in the `mlflow` UI with information on the training run and the model itself.

To find the model, look in the `mlruns` directory at the base of the project:

    ls mlruns/0/<run id given at end of successful run>/artifacts/<model key from train.py>

E.g. For the final model after fine-tuning:

    ls mlruns/0/8237d734f1d94fd893368dd455565f2d/artifacts/keras-yolo-model

It will be called something like `model.h5`.

## Deploying with the Azure Machine Learning Integration

See [AzureML export option and CLI commands](https://mlflow.org/docs/latest/models.html#microsoft-azureml) for the main directions.

For instance, to export to `azureml`-friendly deployment format/structure (and create neccessary files for this deployment type):

    mlflow azureml export -m mlruns/0/8237d734f1d94fd893368dd455565f2d/artifacts/keras-yolo-model -o yolo-output

The `azureml` CLI commands are:

    ```unix
    az ml env setup -l [Azure Region, e.g. eastus2] -n [your environment name] [-g [existing resource group]]
    az ml env set -n [environment name] -g [resource group]
    mlflow azureml deploy <parameters> - to deploy locally to test the model

    az ml env setup -l [Azure Region, e.g. eastus2] -n [your environment name] [-g [existing resource group]]
    az ml env set -n [environment name] -g [resource group]
    mlflow azureml deploy <parameters> - deploy to the cloud
  
    ```

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