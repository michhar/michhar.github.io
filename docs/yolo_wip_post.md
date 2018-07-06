## Notes on Training a Tiny YOLO v3 Model for Use in an iOS App

My System
* Windows 10
* NVIDIA GeForce GTX 1060 Graphics Card
* CUDA 9.0
* cuDNN 7.0

Setup - see https://github.com/AlexeyAB/darknet for details and download links
* MS Visual Studio 2015
* OpenCV 3.4.0

Steps:

1.  Built `Yolo_mark` for Windows according to instructions at https://github.com/AlexeyAB/Yolo_mark (clone and run)
  * Make sure you have OpenCV 3.4.0 at it's at `C:\opencv_3.0\opencv` (Linux instructions on repo Readme)
  * This will produce an executable at `C:\Users\<directory to solution>\Yolo_mark\x64\Release\yolo_mark.exe`
2.  Label images with bouding boxes and classes according to the steps on `Yolo_mark` under "To use for labeling your custom images"
3.  Built `darknet` for Windows according to instructions at https://github.com/AlexeyAB/darknet#how-to-compile-on-windows
4.  Label `jpg`s with the Yolo_mark annotation tool from https://github.com/AlexeyAB/Yolo_mark 
5.  Train a Tiny YOLO v3 model on custom images according to more instructions at https://github.com/AlexeyAB/darknet#how-to-train-tiny-yolo-to-detect-your-custom-objects

What is YOLO

Great articles here:
  * Official fun YOLOv3 [Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  * Great post from an ML expert and iOS App builder, Matthijs Hollemans [Blog](http://machinethink.net/blog/object-detection-with-yolo/)
