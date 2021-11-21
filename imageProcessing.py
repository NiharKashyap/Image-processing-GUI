import cv2
import numpy as np
from skimage import filters
from skimage import color

def save(image,new_image):
	cv2.imwrite('imgO.jpg', image)
	cv2.imwrite('imgS.jpg', new_image)

'''
def saveSegmented(image, new_image):
    cv2.imwrite('imgO.jpg', image)
    cv2.imwrite('imgS.jpg', 255*new_image)
'''

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

def adaptiveThresh(image, maxVal):
	#cv2.adaptiveThreshold(source, maxVal, adaptiveMethod, thresholdType, blocksize, constant)

	R, G, B = cv2.split(image)
	
	new_R = cv2.adaptiveThreshold(R, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
	new_G = cv2.adaptiveThreshold(G, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
	new_B = cv2.adaptiveThreshold(B, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

	new_image = cv2.merge((new_R, new_G, new_B))
	save(image,new_image)

def otsuThresh(image):
	R, G, B = cv2.split(image)
	
	ret,new_R = cv2.threshold(R,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ret,new_G = cv2.threshold(G,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ret,new_B = cv2.threshold(B,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	new_image = cv2.merge((new_R, new_G, new_B))
	save(image,new_image)

def cannyEdge(image):
	new_image = cv2.Canny(image=image, threshold1=100, threshold2=200) # Canny Edge Detection
	save(image,new_image)

#Edge 
def sobelEdge(image):
	new_image = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
	save(image,new_image)

#Region based segmentation
def RegionSegmentation(image):
    #skimage.filters.sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0)
	#image = cv2.imread('tempDir/' + image)
	#image = np.array(image)
	#image = cv2.resize(image, (640, 480))
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edge_sobel = filters.sobel(image)
	save(image, 255*edge_sobel)
 
def WatershedSegmentation(image, threshVal):
    new_image = np.copy(image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,threshVal*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(image,markers)
    new_image[markers == -1] = [255,0,0]
    img2 = color.label2rgb(markers, bg_label=0)
    #cv2.imwrite("new.jpg", 255*img2)
    save(image, 255*img2)

def RandomWalk(image):
    #random_walker (data,labels,beta: int=130,mode: str=str,tol: float=0.001,copy: bool=True,multichannel: bool=False,return_full_prob: bool=False,spacing: __class__=None)
    pass


