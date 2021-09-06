import cv2
import numpy as np



def brightness_enhancement(image, alpha, beta):
	
	new_image = np.zeros(image.shape, image.dtype)
	
	'''
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			for c in range(image.shape[2]):
				new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
	'''

	new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)

def denoising(image, size, h, hColor):

	new_image = cv2.fastNlMeansDenoisingColored(image,None,h,hColor,size,21)

	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)
