---
img:  IMG_3568.JPG
layout: post
title: Working with MLflow - A Reproducible Project Pattern for ML Training and Deployments
comments: true
description: Sample using a new OSS tool, MfFlow, for an entire Keras Object Detection workflow including deployment with AzureML
cover:  /img/flask-post-diagram.png
tags: [python, keras, yolov3, mlflow]
---

**tl;dr**:  MLflow is already powerful, yet simple, even in alpha release.  It' integration with platforms for training and deployment, such as with AzureML, is incredibly helpful.  After all, don't we all want to deploy eventually?

**Posted:**  2019-09-15

## Introduction to MLFlow

Recently, I discovered MLflow on a hunt for a way to track my models and environment, parameters, training and deployments (that was a nice addition) in a single project and a single place.

A few features that caught my eye:

* Run training and deployment from a remote GitHub repo
* Log the environment in which a model was trained (other logging and even a UI for tracking)
* Many deployment options (local and Azure ML among others)

After trying MLflow, I discovered that I liked:

* The flexibility of running the project in several locations (local, [Databricks](https://databricks.com/), remote linux VMs, etc.)
* Error messaging is clear and includes a nice stacktrace
* The deployment was straighforward as the project for AzureML was generated for me with the files I needed


## Get the Model and Data

Feel free to follow along or create your own project.  This is an example of using MLFlow with an existing repo.

1.  Clone the YOLOv3 Keras example repo:  

> `git clone https://github.com/michhar/mlflow-keras-example.git`

2.  cd into model directory:

> `cd mlflow-keras-example/model_data/`

3.  Download the model, data (or create your own) and pointer file to data.
    * Get a Keras-friendly YOLOv3 base model, converted directly from the Tiny YOLOv3 Darknet model (here, the Tiny weights are used - nice for constrained devices):

      > Click to access through Azure Storage [here (34MB)](https://modelsdata.blob.core.windows.net/data/yolov3-tiny.h5), download and place model in `model_data` subdirectory

    * Get some sample data of lego minifigures with helmets and other head gear to train a model to detect what the figurine is wearing on its head, placing the uznipped folder in the `voc` subdirectory.

      > Click to access through from [here (173MB)](https://modelsdata.blob.core.windows.net/data/JPEGImages.zip) or use your own data (according to instructions on this repo [in Step 1](https://github.com/michhar/keras-yolo3#training) - you'll need to label as well).

    * Get the list of images as a small text file with associated bounding boxes and class.

      > Click to access from [here](https://modelsdata.blob.core.windows.net/data/list_master.txt) and place it in the `voc` subdirectory

## Setup for MLflow

Required Python packages:
  * `mlflow`
  * `azure-cli` (if deploying with AzureML)
    * To deploy with `azureml` one will need, also, an Azure Subscription.

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

    mlfow run .

Or if you want to modify a default parameter or two (use `-P` per parameter) like the number of epochs for the transfer learning stage (`frozen_epochs`) and network fine tuning stage (`fine_tune_epochs`) (note you'd use 100s to 1000s of epochs for these in the real world):

    mlflow run . -P frozen_epochs=5 -P fine_tune_epochs=3

Also, you can monitor the run through `tensorboard` which is part of a callback in the `model.fit` method (change logdir as appropriate).

    tensorboard --logdir logs/default

An `mlflow` option for monitoring is with:

    mlflow ui

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

Now, to get the package for the AzureML deployment CLI:

    pip install -r https://aka.ms/az-ml-o16n-cli-requirements-file

(Corresponding to `azure-cli-ml==0.1.0a27.post3` at the time of writing).

See [AzureML export option and CLI commands from mlflow](https://mlflow.org/docs/latest/models.html#microsoft-azureml) for the detailed directions.

For instance, to export to `azureml`-friendly deployment format/structure (and create neccessary files for this deployment type) the command will have the format:

    mlflow azureml export -m mlruns/<run folder>/<run id>/artifacts/<name of mlflow project> -o <name of new folder for azureml>

E.g.:

    mlflow azureml export -m mlruns/0/8237d734f1d94fd893368dd455565f2d/artifacts/keras-yolo-model -o yolo-output

Note, some additional libraries may need to be specified in the generated `score.py`'s `init()` and `run()`, such as `keras` here:

```python
def init():
    global model
    import keras
    model = load_pyfunc("model")

def run(s):
    import keras
    input_df = pd.read_json(s, orient="records")
    return get_jsonable_obj(model.predict(input_df))
```

> Note: not including necessary packages is the most common source of error in deploying with AzureML

Ensure `yolo-output` (or name used for `mlflow` generated file directory) has all necessary packages beyond `mlflow`, namely:

```
numpy==1.14.2
matplotlib==2.2.2
Keras==2.2.2
tensorflow==1.8.0
Pillow==5.1.0
mlflow==0.5.2
```

The `azureml` CLI commands are:

> Note: one may need to register some environment providers in Azure.
> `az provider register -n Microsoft.MachineLearningCompute`
> `az provider register -n Microsoft.ContainerRegistry`
> `az provider register -n Microsoft.ContainerService`

    az login

    az ml env setup -l [Azure Region, e.g. eastus2] -n [your environment name] [-g [existing resource group]]

    az ml env set -n [environment name] -g [resource group]

    mlflow azureml deploy <parameters>

  
See AzureML documentation for more information on the `az ml` commands for deployment or type `az ml -h`.

Use a model management account (or create one).  List them with:

    az ml account modelmanagement list

Set one with:

    az ml account modelmanagement set -g [resource group] -n [model management name]

__For example, the commands used in this project to deploy locally are as follows__

Log in to Azure:

    az login

Set up an environment:

    az ml env setup -l eastus2 -g localyoloenvrg -n localyoloenv

 - This will take a few minutes.

Choose the environment:

    az ml env set -g localyoloenvrg -n localyoloenv

  Deploy the project, now (locally, but linked to a few resources online):

    mlflow azureml deploy --model-path model -n yoloapp123

  When done, clean up by deleting the resource group with:

    az group delete -g localyoloenvrg

## Setting up MLflow on Databricks

Recommendations.  
Set this up and begin with a PyTorch tutorial with Mlflow server to ensure all is well before moving on to code.

Note:  Make sure the versions of packages on Databricks matches the versions of packages on the Linux VM hosting the Mlflow tracking server.


1. Provision Databricks workspace on Azure - Premium tier in WestUS2 (Premium tier for access control so we can manage users)
2. Provision a small, general-purpose Linux VM (Ubuntu 16.04 Server is good) for an ML tracking server to let us monitor, log and save artifacts from training experiments on Databricks:

    * Use Password for authentication
    * Open up traffic to 80 (HTTP) and 22 (SSH)
3. Provision a new Storage Account or create a container for mlflow artifacts in an existing blob Storage Account.
4. Follow directions for setting up the Mlflow tracking server on this new Linux VM at https://docs.azuredatabricks.net/spark/latest/mllib/mlflow.html#mlflow-mlflow-quick-start-notebook under "Set up a Remote MLflow Tracking Server"
    * Use "python3" and "pip3"
    * To get requirements installed (namely `azure-storage`), must first run from the VM:   
    
    `sudo apt-get install python3-pip python3-dev libffi-dev libssl-dev libxml2-dev libxslt1-dev libjpeg8-dev zlib1g-dev`

    * Need to install "flask" to get this server working right:   pip3 install flask
    * Add inbound rule for port 5000 to the network security gateway for this VM (should be in the same resource group as VM) - docs for this at https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal, but just note, an NSG already exists, so just add the rule there
    * When starting the server make sure to start it in background with "&" - it should then be at  your `http://dnsname:5000`
5. Set up for PyTorch and MLFlow by following this tutorial (instructions on Cluster set up there - do GPU with runtime 4.3): https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-pytorch-azure.html
    * Make sure to set the Azure blob storage connection string in Databricks before trying Mlflow server tracking (see https://docs.azuredatabricks.net/spark/latest/mllib/mlflow.html#mlflow-mlflow-quick-start-notebook)
    * If encountering trouble with cluster deployment, try Worker as NC12 and driver as NC12 type VM and ensure you have the proper quotas

Set these in something like ".azurerc" and source it before launching the tracking server:

AZURE_STORAGE_ACCESS_KEY=<keypart==>
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=whatsinaname;AccountKey=lotsofalphanumericsymbols==;EndpointSuffix=core.windows.net"

## Example

See an example MLflow project at https://github.com/michhar/mlflow-keras-example.

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