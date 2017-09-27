---
layout: post
title: "Shapes and Butterflies"
img: chimi_mharris.jpg
date: 2017-09-28 12:55:00 +0800
description: Using opencv and shapely in Python to create polygons and back
tag: [polygons, masks, opencv, shapely, python]
---

A Monarch buttefly courtesy of National Geographic Kids
![monarch butterfly](http://kids.nationalgeographic.com/content/dam/kids/photos/animals/Bugs/H-P/monarch-butterfly-grass.adapt.945.1.jpg)

**tl:dr**:  We use Opencv Python wrapper and Shapely library to create a mask, convert it to some polygons and then back to an image as a mask - noting some interesting properties of opencv and useful tricks.

All of the code below can be found in [this](https://github.com/michhar/python-jupyter-notebooks/blob/master/datatools/DealingWithGeospatialImages.ipynb) Python jupyter notebook.

> Lesson 1: **opencv reads in as BGR and matplotlib reads in a RGB**, just in case that is ever an issue.

I tested this as follows:

```python

img_file = 'monarch.jpg'

# Matplotlib reads as RGB
img_plt = plt.imread(img_file)
plt.imshow(img_plt)

# Read as unchanged so that the transparency is not ignored as it would normally be by default
# Reads as BGR
img_cv2 = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
plt.imshow(img_cv2)

# Convert opencv BGR back to RGB
# See https://www.scivision.co/numpy-image-bgr-to-rgb/ for more conversions
rgb = img_cv2[...,::-1]
plt.imshow(rgb)
```

**With the following results:**

Matplotlib RGB:
![rgb](https://raw.githubusercontent.com/michhar/python-jupyter-notebooks/master/datatools/plt_read_monarch.png)

OpenCV BGR:
![bgr](https://raw.githubusercontent.com/michhar/python-jupyter-notebooks/master/datatools/cv2_read_monarch.png)

OpenCV BGR converted back:
![bgr to rgb](https://raw.githubusercontent.com/michhar/python-jupyter-notebooks/master/datatools/converted_back_rgb_monarch.png)



> TIP:  If you want a more "blocky" polygon representation to save space or memory use the approximation method (add to the `mask_to_polygons` method).

```python
    # Approximate contours for a smaller polygon array to save on memory
    contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
```

I leave it up to you to download [this]() Python jupyter notebook and try using the RGB image for masking and creating Polygons and back.  Do the results change?  Have fun.


