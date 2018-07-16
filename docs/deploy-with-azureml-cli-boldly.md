---
layout: post
title: "Deploying a Machine Learning Model Easily with Azure ML CLI"
img: IMG_3455.JPG
date: 2018-03-11 12:55:00 +0000
description: 
tag: [azureml, deploy, machine-learning, cntk]
comments: true
---

**Posted:**  2018-03-11

## Context

I was looking for an easy way to deploy a machine learning model I'd trained for classification, built with Microsoft Cogntive Toolkit (CNTK), a deep learning framework.  I wanted still to test locally with the Python code I wrote, then dockerize and test my image locally, as well.  If the local image ran, I wished to, then, deploy the tested, dockerized service to a cluster for a realtime scoring endpoint (with just a handful of commands if possible - and, indeed, it was).

This post is mainly about the commands to use for deploying with the new, in Preview, Azure ML CLI, however for example scoring files and schema with CNTK, see the [References](#references) below.

## Prerequisites

1.  AzureML CLI (Install Using the CLI in this [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#using-the-cli))
2.  Docker installed (for local service testing) - [Ref](https://docs.docker.com/get-started/)
3.  A scoring script (see [References](#references) for examples)
4.  Any other necessary files like labels or necessary `pip` installs in a `requirements.txt`

Note:  The following was all done in Jupyter on a Linux Ubuntu Data Science Virutal Machine ([Doc](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro))

## Overview

Save for writing the actual code and installing the prerequisites, all can be done with the CLI, even running a quick service test call.

The general story goes as follows.  There is a local testing grounds and a remote cluster deployment in the overview outline.

### To Begin

1.  Write a scoring script that has `run` and `init` methods, with a `main` method to make the service payload schema (an example [Ref](https://github.com/Azure/MachineLearningSamples-ImageClassificationUsingCntk/blob/master/scripts/deploymain.py)).  The scoring script is packaged up for use later, but has a dual purpose of generating a schema for the service.  Run this script to generate the schema.  Package and deploy this script to make prediction service.
2.  Write a conda dependencies and/or `pip` install requirements file (this will have the reference to the CNTK wheel to install cntk into the docker image - we'll talk about in a second)
3.  Register three Environment Providers (for the cluster deployment)
5.  Create a Model Management Account in Azure

- There's an option to use one master, do-it-all, command or run separate commands as done here.  The separate commands perform the following.

### For a Local Deployment test (always a good idea)

1.  Set up the local Environment in Azure and switch to it
2.  Register a model (the ML model, e.g. a saved CNTK model in a [Protobuf](https://en.wikipedia.org/wiki/Protocol_Buffers) based format)
3.  Create a manifest for all requirements to build an image (e.g. model, dependencies and can include multiple models)
4.  Create a docker image with the environment and pertinent files
5.  Create and deploy the service using the docker image

### For the Remote Cluster deployment

1.  Set up the remote cluster Environment in Azure and switch to it
2.  Create and deploy the service with the same image as from local deployment

## The Command Sequence

After creating the scoring file, `score.py` here, and placing all necessary package installs into a `requirements.txt` file (for a Python package manager to use) we can begin our deployment.


### Deploy locally to test

For most of the commands as reference see this [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#4-register-a-model), however some more specific instructions are here that may be useful for a specialized framework model as can be made with CNTK or TensorFlow, for example.  Other commands around setup are found in this [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration).

These commands can be run in a Jupyter notebook, hence the bang, "!", preceding the command.  If these are run on the command line please remove the "!".

> TIP:  If on the Data Science Virtual Machine which has this CLI, you may need to run these commands a little differently (replace `az` with `{sys.executable} -m azure.cli`).  e.g. in a Jupyter code cell:
> ```
> # Get Azure ML CLI help on DSVM
> import sys
> ! {sys.executable} -m azure.cli ml -h
> ```

**Log into Azure**

Simply:

 ```bash
# This will send a code to prompt for password through the browser
! az login
```

Or if you've created a couple of system variables with the username and password:

```bash
# Using two system variable, the user-defined Azure username and password
! az login --username "$AZ_USER" --password "$AZ_PASS"
```


**Register three Environment Providers**

To start the setup process, you need to register a few environment providers by entering the following commands

```bash
! az provider register -n Microsoft.MachineLearningCompute
! az provider register -n Microsoft.ContainerRegistry
! az provider register -n Microsoft.ContainerService
```

**Create a Model Management Account in Azure**

```bash
# ! az ml account modelmanagement create -l [Azure region, e.g. eastus2] -n [your account name] -g [resource group name] --sku-instances [number of instances, e.g. 1] --sku-name [Pricing tier for example S1]
! az ml account modelmanagement create -l westeurope -n happymodelmgmt -g happyprojrg --sku-instances 1 --sku-name S1
```

```bash
# az ml account modelmanagement set -n [your account name] -g [resource group it was created in]
! az ml account modelmanagement set -n happymodelmgmt -g happyprojrg
```

**Set up the local Environment in Azure and switch to it**

```bash
# az ml env setup -l [Azure Region, e.g. eastus2] -n [your environment name] [-g [existing resource group]]
! printf 'y' | az ml env setup -l "West Europe" -n localenv -g happyprojrg
```


```bash
# az ml env show -n [environment name] -g [resource group]
! az ml env show -n localenv -g happyprojrg
```

```bash
# az ml env set -n [environment name] -g [resource group]
! az ml env set -n localenv -g happyprojrg
```


**Register a model  (the ML model, e.g. a saved CNTK model in a [Protobuf](https://en.wikipedia.org/wiki/Protocol_Buffers) based format)**

```bash
# Get help on this
! az ml model register --help
```

This will output the model ID:
```bash
# az ml model register --model [path to model file] --name [model name]
! az ml model register --model happy_classifier_cntk.model --name happy_classifier_cntk.registered.model
```

```bash
# Show the registered models
! az ml model list -o table
```

**Create a manifest for all requirements to build an image**

```bash
# Get help on this
! az ml manifest create --help
```

After having the requirements file (user generated list of `pip` installable packages needed) and the service schema file (representing the json payload for the service call which is created by running the `main` method in `score.py` mentioned above, e.g., `python score.py`), one can create the manifest to hold this information along with other requirements.

```bash
# az ml manifest create --manifest-name [your new manifest name] --model-id [model id] -f [path to code file] -r [runtime for the image, e.g. spark-py] -p [pip installs, e.g. requirements.txt] -d [extra files, e.g. a label file] -s [service schema. e.g. service_schema.json] --verbose --debug
# Note must have requirements file and manifest name mustn't have underscores but rather '.' or '-'
! az ml manifest create --manifest-name happyclassifiermanifest --model-id [model id from register command] -r python -p requirements.txt -d target_set.txt -f score.py -s service_schema.json #--verbose --debug
```

```bash
! az ml manifest show -i [manifest id]
```

**Create a docker image with the environment and pertinent files**

```bash
# Ensure correct permissions for docker and add user to docker group
! sudo chmod -R ugo+rwx /var/run/
! sudo usermod -aG docker [your current user]
```

```bash
# Get help on this
! az ml image create --help
```

This will produce an image ID:
```bash
# az ml image create -n [image name] --manifest-id [the manifest ID]
! az ml image create -n happyclassifierimage --manifest-id [manifest id]
```

```bash
# Get the usage in order to pull the image
! az ml image usage -i [image id]
```

**(Optional) Test the docker image**

```bash
# To log in as a docker user
! az ml env get-credentials -g happyprojrg -n localenv
```

```bash
# Log in to docker and pull down image from ACR, then run
! docker login -u [username] -p [password] [loginServer]
! docker pull [image name from usage command]
! docker run [image name from usage command]
```

**Create and deploy the service using the docker image**

```bash
# az ml service create realtime --image-id [image id] -n [service name]
! printf 'y' | az ml service create realtime --image-id [image id] -n localhappywebservice
```


```bash
# Get logs from creation in case something went wrong
! az ml service logs realtime -i localhappywebservice
```

```bash
# az ml service usage realtime -i [service id]
! az ml service usage realtime -i localhappywebservice
```

**Test it**

```bash
# az ml service run realtime -i <service id> -d "{\"input_df\": [{\"sepal length\": 3.0, \"sepal width\": 3.6, \"petal width\": 1.3, \"petal length\":0.25}]}"
# Note the removal in the json payload of the "u" or unicode designation from docs
! az ml service run realtime -i localhappywebservice -d "{\"request_package\": {\"url\": \"https://contents.mediadecathlon.com/p350121/2000x2000/sq/mountaineering_boots_-_blue_standard_sizes41_42_43_44_45_46_simond_8324356_350121.jpg?k=362304aaf6fecd4b2c8750987a2fb104\"}}"
```

### Deploy to a cluster

Follow this [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#cluster-deployment) for more information on cluster deployment.  Below are the pertinent commands working at of 2018-03-11.

**Set up the remote cluster Environment in Azure and switch to it**

```bash
# az ml env setup --cluster -n [your environment name] -l [Azure region e.g. eastus2] [-g [resource group]]
! printf 'n\nY' | az ml env setup --cluster -n clusterenv -l westeurope -g happyprojrg
```

```bash
# Is the environment ready?
! az ml env show -g happyprojrg -n clusterenv
```

```bash
# Set the environment to the remote cluster
! az ml env set -g happyprojrg -n clusterenv
```

```bash
# Set to same model management account as local
! az ml account modelmanagement set -n happymodelmgmt -g happyprojrg
```

**Create and deploy the service with the same image as from local deployment**

```bash
# One command to do it all from "scratch"
# ! az ml service create realtime --model-file happy_classifier_cntk.model -f score.py -n remotehappywebservice -s service_schema.json -r python -p requirements.txt -d target_set.txt --verbose --debug
```

```bash
# Remotely deployed kubernetes cluster for predicting and scoring new images with the model 
! az ml service create realtime --image-id [image id] -n remotehappywebservice
```

**Test it**

```bash
! az ml service run realtime -i [service id] -d "{\"request_package\": {\"url\": \"https://contents.mediadecathlon.com/p350121/2000x2000/sq/mountaineering_boots_-_blue_standard_sizes41_42_43_44_45_46_simond_8324356_350121.jpg?k=362304aaf6fecd4b2c8750987a2fb104\"}}"
```

```bash
# Even though successful, might still take some time to deprovision everything in Azure
! az ml service delete realtime --id [service id]
```

Finis!

## References

**Important Notes**

* There are different input data type options for sending up to the service and you can specify this when you generate the schema for the service call.
* Install the Azure ML CLI into the system Python if using a DSVM and the main Python in a local setup with (from this [Doc](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration#using-the-cli)):
    `! sudo pip install -r https://aka.ms/az-ml-o16n-cli-requirements-file`
* When creating the image with the `az ml` cli, remember to include all files necessary with the `-d` flag such as any label or data files.  Avoid using the `-c` flag for the conda dependencies file for the time being.  If particluar installs are needed, a `requirements.txt` file can be used with the `pip` installable packages specified and this files should go after a`-p` flag.

**Overview**

* Overview of Azure ML model management <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-overview" target="_blank">Doc</a>
* Deployment walkthrough <a href="https://azure.github.io/LearnAI-Bootcamp/lab04.2-deploying_a_scoring_service_to_aks/0_README" target="_blank">Ref</a>

**More on Deployment**

* Microsoft Blog on deploying from Azure ML Workbench and the Azure ML CLI <a href="https://blogs.technet.microsoft.com/machinelearning/2017/09/25/deploying-machine-learning-models-using-azure-machine-learning/" target="_blank">Ref</a>
* Setting up with the Azure ML CLI for deployment 
<a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration" target="_blank">Doc</a>
* Non-CLI deployment methods (AML alternative) <a href="https://github.com/Azure/ACS-Deployment-Tutorial" target="_blank">Ref</a>

**Scoring File and Schema Creation References**

* Example of schema generation <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#2-create-a-schemajson-file" target="_blank">Doc</a>
* Example of the scoring file showing a CNTK model and serializing an image as a `PANDAS` data type for input data to service <a href="https://github.com/Azure/MachineLearningSamples-ImageClassificationUsingCntk/blob/master/scripts/deploymain.py" target="_blank">Ref</a>
* Example of the scoring file showing a `scikit-learn` model and a `STANDARD` data type (json) for input data to service <a href="https://github.com/Azure/Machine-Learning-Operationalization/blob/master/samples/python/code/newsgroup/score.py" target="_blank">Ref</a>
* After creating a `run` and `init` methods as in the links above, plus a schema file, begin with "Register a model" found in this <a href="https://docs.microsoft.com/en-us/azure/machine-learning/preview/model-management-service-deploy#4-register-a-model">Doc</a>

**Docker**

* Docker Docs <a href="https://docs.docker.com/get-started/" target="_blank">Ref</a>


    