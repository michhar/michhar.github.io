# Notes on Training a Tiny YOLO v3 Model for Use in an iOS App

## Pipeline for Creating and Using a Darknet-Based YOLOv3 Model

1.  Label and train with Darknet (Windows 10 and NVIDIA GeForce GPU)
2.  Convert Darknet model to Keras (macOS) - `keras-yolo3` project
3.  Convert Keras model to CoreML (macOS) - `coremltools` package is mac OS only as of writing this
4.  Run iOS XCode project and build to iPhone/iPad (macOS/iOS)

Convert to Keras with script from https://github.com/qqwweee/keras-yolo3

    python convert.py experiment/yolov3-tiny.cfg model_data/yolov3-tiny_lr0.001_July7.weights model_data/yolov3-tiny-minifig.h5

CoreML in https://github.com/Ma-Dan/YOLOv3-CoreML/tree/master/Convert
  * Update output??

In https://github.com/hollance/YOLO-CoreML-MPSNNGraph XCode project
  * Update `labels` in `Helpers.swift`
  * Update `numClasses` in `YOLO.swift`
  * Add the `TinyYOLO.mlmodel` from the custom training/conversion above
  
## macOS Instructions for Training with Darknet (CPU)

### My System
* macOS High Sierra 10.13.5
* MacBook Pro (15-inch, 2017)

### Steps

1.  Build `Yolo_mark` for Linux according to instructions at https://github.com/AlexeyAB/Yolo_mark (clone and run)

2. Build `darknet` with opencv (mostly following instructions at https://pjreddie.com/darknet/install/):

    2.1  `brew install opencv@2`
    
    2.2  Clone `darknet` repo:  `git clone https://github.com/pjreddie/darknet`
    
    2.3  `cd darknet`
    
    2.4  In the `Makefile` update the opencv parameter to `OPENCV=1` (change any other pertinent parameters as well here)
    
    2.5  Run `make` in the base `darknet` directory

3. Label images with bouding boxes and classes according to the steps on `Yolo_mark` under "To use for labeling your custom images"

4. Train a Tiny YOLO v3 model on custom images according to more instructions at https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects

    4.1 Note, put your `obj.data` file in the `cfg` folder and use full paths to the `train.txt` and `obj.names`

    4.2 If doing transfer learning, place `stopbackward=1` in the `yolov3-tiny.cfg`

    4.3 Run with:

        ./darknet detector train cfg/lego.data experiment/yolov3-tiny.cfg experiment/yolov3-tiny.conv.15
    
    4.4 The final trained model with be in the `backup` folder.

## Windows Instructions for Training with Darknet (GPU)

> TIP:  when moving any files from Windows to macOS, check for proper newlines (e.g. lack of `^M` characters in text files)

### My System
* Windows 10
* NVIDIA GeForce GTX 1060 Graphics Card
* CUDA 9.1
* cuDNN 7.0

### Steps (WIP)

Setup - see https://github.com/AlexeyAB/darknet for details and download links
* MS Visual Studio 2017 (instructions say 2015, but 2017 worked well for me)
* OpenCV 3.4.0

1.  Built `Yolo_mark` for Windows according to instructions at https://github.com/AlexeyAB/Yolo_mark (clone and run)
  * Make sure you have OpenCV 3.4.0 at it's at `C:\opencv_3.0\opencv` (Linux instructions on repo Readme)
  * This will produce an executable at `C:\Users\<directory to solution>\Yolo_mark\x64\Release\yolo_mark.exe`
  * Find files `opencv_world320.dll` and `opencv_ffmpeg320_64.dll` (or `opencv_world340.dll` and `opencv_ffmpeg340_64.dll`) in `C:\opencv_3.0\opencv\build\x64\vc14\bin` and put it near with `yolo_mark.exe`
2.  Label images with bouding boxes and classes according to the steps on `Yolo_mark` under "To use for labeling your custom images"
3.  Built `darknet` for Windows according to instructions at https://github.com/AlexeyAB/darknet#how-to-compile-on-windows
  * Except, use VS 2017 with **Platform Toolset `Visual Studio 2015 (v140)`** (right-click solution -> Properties -> Configuration Properties (General))
  * Make sure `cudnn.h` is in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include` folder
  * Make sure `cudnn.lib` is in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64` folder
  > TIP:  when installing the NVIDIA CUDA Toolkit 9.1, you may need to do an Advanced/Custom install and "uncheck" Visual Studio Integration as this may cause install not to work properly
5.  Train a Tiny YOLO v3 model on custom images according to more instructions at https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects
  * May need to
    * Place certain cuDNN libs into v9.1 CUDA directory 
      (e.g. `cudnn64_7.dll` found in search after a cuDNN intall into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin`)
  * Run with:
  
      `darknet.exe detector train cfg\lego.data experiment\yolov3-tiny.cfg experiment\yolov3-tiny.conv.15`
      
  * The final trained model with be in the `backup` folder.

## What is YOLO and Object Detection

Great Series of Videos on Object Detection and YOLO (Convolutional Neural Network (CNNs) by Andrew Ng [Full Course]):
  * Object Detection is CNN20-31, with YOLO being CNN31 ([Videos](https://www.youtube.com/watch?v=Z91YCMvxdo0&list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud))

Great articles here:
  * Official fun YOLOv3 [Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  * Great post from an ML expert and iOS App builder, Matthijs Hollemans [Blog](http://machinethink.net/blog/object-detection-with-yolo/)
