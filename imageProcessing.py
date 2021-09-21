import cv2
import numpy as np



def brightness_enhancement(image, alpha, beta):
	
	new_image = np.zeros(image.shape, image.dtype)
	new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)

def denoising(image, size, h, hColor):

	new_image = cv2.fastNlMeansDenoisingColored(image,None,h,hColor,size,21)

	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)
