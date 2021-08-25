import numpy as np
import matplotlib.pyplot as plt
import cv2


eye_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")


img=cv2.imread("Hermione1.jpg")



img1=img.copy()

eye=eye_cascade.detectMultiScale(img)[0]
print (eye)


eye_x,eye_y,eye_w, eye_h= eye
img = cv2.rectangle(img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255,255,255), 5 )
# cv2.imshow('',img)
# cv2.waitKey(0)



glasses=cv2.imread("sample1.png")



glasses=cv2.resize(glasses,(eye_w+50,eye_h+55))



for i in range(glasses.shape[0]):
  for j in range(glasses.shape[1]):
  	print(glasses)
  	if (glasses[i,j,3]>0):
  		img1[eye_y+i-20,eye_x+j-23, :]=glasses[i,j,:-1]
'''

x_offset = eye_x
y_offset = eye_y


x_end = x_offset + glasses.shape[1]
y_end = y_offset + glasses.shape[0]


img1[y_offset:y_end,x_offset:x_end] = glasses
'''

cv2.imshow('',img1)
cv2.waitKey(0)