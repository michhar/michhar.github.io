import cv2
import glob

for img in glob.glob('../assets/img/*'):
    im = cv2.imread(img)
    if im != None:
        im = cv2.resize(im, (600, 450))
        cv2.imwrite(img, im)