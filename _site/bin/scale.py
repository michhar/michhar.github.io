import cv2
import glob

files = glob.glob("**/*.png", recursive=True)

for f in files:
    try:
        im = cv2.imread(f)
        im = cv2.resize(im, (675, 450))
        cv2.imwrite(f, im)
    except Exception as e:
        pass


