import cv2
import numpy as np
from skimage import filters
from skimage import color
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import random_walker
from sklearn.cluster import KMeans
from scipy import ndimage



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

def prewittEdge(image):
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(image, -1, kernelx)
	img_prewitty = cv2.filter2D(image, -1, kernely)
	new_image = np.sqrt(pow(img_prewittx, 2.0) + pow(img_prewitty, 2.0))
	save(image,new_image)


def logEdge(image):
	# Apply gaussian blur
	blur_img = cv2.GaussianBlur(image, (5, 5), 0)
	# Positive Laplacian Operator
	new_image = cv2.Laplacian(blur_img, cv2.CV_64F)
	new_image = new_image*255
	save(image,new_image)

def robertsEdge(image):
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	kernelx = np.array([[1, 0], [0, -1]])
	kernely = np.array([[0, 1], [-1, 0]])

	img_robertx = cv2.filter2D(image, -1, kernelx)
	img_roberty = cv2.filter2D(image, -1, kernely)
	new_image = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
	new_image=new_image*255
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
 
def WatershedSegmentation(image, threshVal, iteration, maskSize):
    new_image = np.copy(image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = iteration)
    
    sure_bg = cv2.dilate(opening,kernel,iterations=iteration)
    
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,maskSize)
    ret, sure_fg = cv2.threshold(dist_transform,threshVal*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(image,markers)
    new_image[markers == -1] = [255,0,0]
    img2 = color.label2rgb(markers, bg_label=0)
    save(image, 255*img2)

def circle_points(resolution, center, radius):
    """
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)
    c = center[1] + radius*np.cos(radians)#polar co-ordinates
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T

def ActiveContour(image, resolution, center, radius):
	#points = circle_points(resolution, [80, 250], 80)[:-1]
	pass

def RandomWalk(image):
    #random_walker (data,labels,beta: int=130,mode: str=str,tol: float=0.001,copy: bool=True,multichannel: bool=False,return_full_prob: bool=False,spacing: __class__=None)
	markers = np.zeros(image.shape, dtype=np.uint)
	markers[image < -0.95] = 1
	markers[image > 0.95] = 2

	# Run random walker algorithm
	labels = random_walker(image, markers, beta=10, mode='bf')
	save(image, labels)

def clusterSeg(img, cluster):
	# For clustering the image using k-means, we first need to convert it into a 2-dimensional array
	image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
	# tweak the cluster size and see what happens to the Output
	kmeans = KMeans(n_clusters=cluster, random_state=0).fit(image_2D)
	clustered = kmeans.cluster_centers_[kmeans.labels_]
	# Reshape back the image from 2D to 3D image
	clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
	save(img, clustered_3D)


def CircularHough(image, minDist, param1, param2, minRad, maxRad):
	new_image = np.copy(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
	# Blur using 3 * 3 kernel.
	gray_blurred = cv2.blur(gray, (3, 3))
	  
	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(gray_blurred, 
	                   cv2.HOUGH_GRADIENT, 1, minDist, param1,
	               param2, minRad, maxRad)
	  
	# Draw circles that are detected.
	if detected_circles is not None:
	  
	    # Convert the circle parameters a, b and r to integers.
	    detected_circles = np.uint16(np.around(detected_circles))
	  
	    for pt in detected_circles[0, :]:
	        a, b, r = pt[0], pt[1], pt[2]
	  
	        # Draw the circumference of the circle.
	        cv2.circle(new_image, (a, b), r, (0, 255, 0), 2)
	  
	        # Draw a small circle (of radius 1) to show the center.
	        cv2.circle(new_image, (a, b), 1, (0, 0, 255), 3)

	save(image, new_image)

def EllipseHough(image):
	pass