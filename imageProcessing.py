import cv2
import numpy as np


def save(image,new_image):
	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)

def sharpen(image):
	# Create our shapening kernel, it must equal to one eventually
	kernel_sharpening = np.array([[-1,-1,-1], 
	                              [-1, 9,-1],
	                              [-1,-1,-1]])
	# applying the sharpening kernel to the input image & displaying it.
	new_image = cv2.filter2D(image, -1, kernel_sharpening)
	save(image,new_image)
	

def brightness_enhancement(image, alpha, beta):
	
	new_image = np.zeros(image.shape, image.dtype)
	new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	save(image,new_image)

def denoising(image, size, h, hColor):

	new_image = cv2.fastNlMeansDenoisingColored(image,None,h,hColor,size,21)

	save(image,new_image)

def blur(image, kernel):
	
	new_image = cv2.blur(image, (kernel,kernel))
	save(image,new_image)

def gauss(image, kernel):
	new_image = cv2.GaussianBlur(image, (kernel,kernel), 0)
	save(image,new_image)

def median(image,kernel):
	new_image = cv2.medianBlur(image, kernel)
	save(image,new_image)


def bilateral(image,d,sigmaColor,sigmaSpace):
	new_image = cv2.bilateralFilter(image,d,sigmaColor,sigmaSpace)
	save(image,new_image)

def equi_hist(image):

	R, G, B = cv2.split(image)

	output1_R = cv2.equalizeHist(R)
	output1_G = cv2.equalizeHist(G)
	output1_B = cv2.equalizeHist(B)

	new_image = cv2.merge((output1_R, output1_G, output1_B))
	save(image,new_image)

def clahe(image, clipLim):
	clahe=cv2.createCLAHE(clipLim)
	R, G, B = cv2.split(image)

	output1_R = clahe.apply(R)
	output1_G = clahe.apply(G)
	output1_B = clahe.apply(B)

	new_image = cv2.merge((output1_R, output1_G, output1_B))
	save(image,new_image)

def globalThresh(image, thresh, maxVal):
	th, new_image = cv2.threshold(image, thresh, maxVal, cv2.THRESH_BINARY);
	save(image,new_image)



