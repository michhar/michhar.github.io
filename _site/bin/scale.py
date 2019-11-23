import cv2
import glob

files = glob.glob("*.jpg", recursive=False)
files_more = glob.glob("*.JPG", recursive=False)
files.extend(files_more)

for f in files:
    try:
        print(f)
        im = cv2.imread(f)
        print(im.shape)
        im = cv2.resize(im, (600, 450))
        print(im.shape)
        cv2.imwrite(f, im)
    except Exception as e:
        pass


