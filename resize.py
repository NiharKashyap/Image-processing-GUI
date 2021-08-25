import cv2
import os

for root, subdirs, files in os.walk('C:/Code/Projects/Image Processing GUI/BackgroundImages'):
    for f in files:
        if f.endswith('jpg'):
            # print(f)
            img = cv2.imread('C:/Code/Projects/Image Processing GUI/BackgroundImages/' + f)
            img = cv2.resize(img, (640, 480))
            cv2.imwrite('C:/Code/Projects/Image Processing GUI/BackgroundImages/'+f, img)
            print(*["Image", f, "is resized to 640 X 480"])