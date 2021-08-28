import cv2
#import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


def rem(frame, path, threshold):
    segmentor = SelfiSegmentation()
    #fpsReader = cvzone.FPS()

    img = cv2.imread(path)
    
    imgOut = segmentor.removeBG(frame, img, threshold)
    cv2.imwrite('imgO.jpg', frame)
    cv2.imwrite('imgS.jpg', imgOut)
    #imgStack = cvzone.stackImages([frame, imgOut], 2,1)
    #_, imgStack = fpsReader.update(imgStack)
    #cv2.imwrite('img.jpg', imgStack)
