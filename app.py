import cv2
import streamlit as st
from PIL import Image
import imageProcessing as ip
import numpy as np
import os

st.set_page_config(layout='wide')
#left,center = st.columns([3,8])

hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


choice = st.sidebar.selectbox('Select Your Application', 
	('Image Pre Processing', 'Image Segmentation', 'Comparision'))


uploaded_file = st.file_uploader("Choose a Image file")

Oimage, Simage = st.columns(2)

def preProcess(uploaded_file):
	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		
		my_img = cv2.imread('tempDir/' + uploaded_file.name)
		frame = np.array(my_img)
		frame = cv2.resize(frame, (640, 480))
		return frame

def preProcessRegion(uploaded_file):
	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		
	return uploaded_file.name
    	

def paintImage():
	imgO = Image.open('imgO.jpg')
	imgS = Image.open('imgS.jpg')
	Oimage.image(imgO)
	Oimage.header("Original")
	Simage.image(imgS)
	Simage.header("Processed")
	with open("imgS.jpg", "rb") as file:
		Simage.download_button(label="Download image",
             data=file,
             file_name="Processed.jpg",
             mime="image/png"
           )

def paintImageComp(flag):
	
	imgO = Image.open(str(flag)+'.jpg')

	if flag==1:
		Oimage.image(imgO)
	elif flag==2:
		Simage.image(imgO)


def paintDenoise():
	choice = st.sidebar.selectbox('Select Algorithm', ('Image Blur', 'Gaussian Filter',
		'Median Filter', 'Bilateral Filter', 'Non Local Means'))

	if choice=='Image Blur':
		thresh = st.sidebar.slider('Kernel Size', 0, 20, 6, help='We travel through the image with this filter by applying the desired operation.')
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.blur(frame, thresh)
			paintImage()


	elif choice=='Gaussian Filter':
		thresh = st.sidebar.slider('Kernel Size', 1, 21, 5, step=2, help='We travel through the image with this filter by applying the desired operation.')
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.gauss(frame, thresh)
			paintImage()

	elif choice=='Median Filter':
		thresh = st.sidebar.slider('Kernel Size', 1, 21, 5,step=2, help='We travel through the image with this filter by applying the desired operation.')
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.median(frame, thresh)
			paintImage()

	elif choice=='Bilateral Filter':
		d = st.sidebar.slider('Kernel Size', 0, 20, 6, help='Diameter of each pixel neighborhood.')
		sigmaColor = st.sidebar.slider('Sigma Color', 0, 100, 75, help='The greater the value, the colors farther to each other will start to get mixed')
		sigmaSpace = st.sidebar.slider('Sigma Space', 0, 100, 75, help='The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range')

		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.bilateral(frame, d, sigmaColor,sigmaSpace)
			paintImage()


	elif choice=='Non Local Means':
		level = st.sidebar.slider('Window size', 1, 10, 2, help='Size in pixels of the template patch.')
		h = st.sidebar.slider('H', 1, 20, 2, help='parameter deciding filter strength. Higher h value removes noise better, but removes details of image also')
		hColor = st.sidebar.slider('HColor', 1, 20, 2, help='same as h, but for color images only')
		frame = preProcess(uploaded_file)
		ip.denoising(frame, level, h, hColor)
		paintImage()


def paintContrast():
	#Removing Min Max COntrast STretrching for now
	choice = st.sidebar.selectbox('Select Algorithm', ('Histogram Equilization', 
		'CLAHE'))

	if choice=='Histogram Equilization':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.equi_hist(frame)
			paintImage()
		
	elif choice=='CLAHE':
		clipLim = st.sidebar.slider('Clip Size', 1,100,15, help='Limits the amplification by clipping the histogram at a predefined value')
		tileGridSize=st.sidebar.slider('Tile Grid Size', 1,10,8, help='Image is divided into small blocks called tiles')
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.clahe(frame, clipLim, tileGridSize)
			paintImage()

def paintBrightness():
	alpha = st.sidebar.slider('Contrast', 1.0, 3.0, 1.25)
	beta = st.sidebar.slider('Brightness', 0, 100, 25)

	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.brightness_enhancement(frame, alpha, beta)
		paintImage()


def paintSharp():
	
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.sharpen(frame)
		paintImage()


def paintThresh():
	#Removing Local for now
	choice = st.sidebar.selectbox('Select Algorithm', ('Global', 
		 'Adaptive', 'Otsu'))

	if choice=='Global':
		thresh = st.sidebar.slider('Threshold Value', 0, 100, 127)
		maxVal = st.sidebar.slider('Max Value', 0, 100, 127, help='Maximum value to be used with THRESH_BINARY ')
		
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.globalThresh(frame, thresh, maxVal)
			paintImage()

	elif choice=='Adaptive':
		maxVal = st.sidebar.slider('Max Value', 0, 100, 127, help='maximum value to be used with THRESH_BINARY')
		
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.adaptiveThresh(frame, maxVal)
			paintImage()

	elif choice=='Otsu':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.otsuThresh(frame)
			paintImage()



def paintEdge():
	st.title('Edge Detection')
	choice = st.sidebar.selectbox('Select Algorithm', ('Canny', 
		'Sobel', 'Prewitt', 'LoG', 'Roberts'))

	if choice=='Canny':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.cannyEdge(frame)
			paintImage()

	elif choice=='Sobel':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.sobelEdge(frame)
			paintImage()

	elif choice=='Prewitt':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.prewittEdge(frame)
			paintImage()

	elif choice=='LoG':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.logEdge(frame)
			paintImage()

	elif choice=='Roberts':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.robertsEdge(frame)
			paintImage()


def paintReg():
	st.title('Region based Segmentation')
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.RegionSegmentation(frame, 0)
		paintImage()



def paintShaped():
	#Removing Ellipse
	choice = st.sidebar.selectbox('Select Algorithm', ('Circular', 
		))
	
	if choice=='Circular':
		minDist = st.sidebar.slider('Minimum Distance', 0, 200, 32, help='Minimum distance between two circles.')
		#param1 = st.sidebar.slider('Param 1', 0.0, 1.0, 0.3, help='Minimum distance between two circles.')
		#param2 = st.sidebar.slider('Param 2', 0.0, 1.0, 0.3, help='Minimum distance between two circles.')
		param1=0.3
		param2=0.3
		minRad = st.sidebar.slider('Min Radius', 0, 200, 22, help='Minimum distance between two circles.')
		maxRad = st.sidebar.slider('Max Radius', 0, 200, 22, help='Maximum distance between two circles.')

		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.CircularHough(frame, minDist, param1, param2, minRad, maxRad)
			paintImage()

	elif choice=='Ellipse':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.EllipseHough(frame)
			paintImage()

def paintWater():
	st.title('Watershed based Segmentation')
	thresh = st.sidebar.slider('Threshold Value', 0.0, 1.0, 0.2)
	iterations = st.sidebar.slider('Number of iterations', 0, 10, 2)
	maskSize = st.sidebar.selectbox('Mask Size', (0, 3, 5), help='Size of mask')

	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.WatershedSegmentation(frame, thresh, iterations, maskSize, 0)
		paintImage()



def paintActive():
	st.title('Active Contour model based Segmentation')

def paintRandomWalk():
	st.title('Random walker segmentation')
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.RandomWalk(frame)
		paintImage()


def paintCluster():
	st.title('Cluster based segmentation')
	cluster = st.sidebar.slider('Clusters', 0, 10, 2, help='Number of clusters.')
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.clusterSeg(frame, cluster, 0)
		paintImage()

def paintClusterComp(flag):
	cluster = st.sidebar.slider('Clusters', 0, 10, 2, help='Number of clusters.')
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.clusterSeg(frame, cluster, flag)
		paintImageComp(flag)

def paintWaterComp(flag):
	
	thresh = st.sidebar.slider('Threshold Value', 0.0, 1.0, 0.2)
	iterations = st.sidebar.slider('Number of iterations', 0, 10, 2)
	maskSize = st.sidebar.selectbox('Mask Size', (0, 3, 5), help='Size of mask')

	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.WatershedSegmentation(frame, thresh, iterations, maskSize, flag)
		paintImageComp(flag)


def paintRegComp(flag):
	
	frame = preProcess(uploaded_file)
	if frame is not None:
		ip.RegionSegmentation(frame, flag)
		paintImageComp(flag)

def compare(choice1,choice2):
	pass

def painter(id):
	if id==1:
		choice = st.sidebar.selectbox('Select PreProcessing Technique',
		('Denoising', 'Contrast Enhancement', 'Brightness Enhancement',
		'Sharpness Enhancement'))

		if choice=='Denoising':
			paintDenoise()
		elif choice=='Contrast Enhancement':
			paintContrast()
		elif choice=='Brightness Enhancement':
			paintBrightness()
		elif choice=='Sharpness Enhancement':
			paintSharp()
	
	elif id==2:
		#Removed Active COntour and Random Walker
		choice = st.sidebar.selectbox('Select Segmentation Technique',
		('Thresholding', 'Edge Detection', 'Region based Segmentation', 'Regular shaped object segmentation',
		'Watershed based Segmentation','Cluster based segmentation'))

		if choice=='Thresholding':
			paintThresh()
		elif choice=='Edge Detection':
			paintEdge()
		elif choice=='Region based Segmentation':
			paintReg()
		elif choice=='Regular shaped object segmentation':
			paintShaped()
		elif choice=='Watershed based Segmentation':
			paintWater()
		elif choice=='Active Contour model based Segmentation':
			paintActive()
		elif choice=='Random walker segmentation':
			paintRandomWalk()
		elif choice=='Cluster based segmentation':
			paintCluster()

	elif id==3:
		choice1 = st.sidebar.selectbox('Select 1st Segmentation Technique',
		('Region based Segmentation','Watershed based Segmentation',  'Cluster based segmentation'))

		if choice1=='Region based Segmentation':
			paintRegComp(1)
		elif choice1=='Watershed based Segmentation':
			paintWaterComp(1)
		elif choice1=='Cluster based segmentation':
			paintClusterComp(1)



		choice2 = st.sidebar.selectbox('Select 2nd Segmentation Technique',
		('Region based Segmentation','Watershed based Segmentation', 'Cluster based segmentation'))

		if choice2=='Region based Segmentation':
			paintRegComp(2)
		elif choice2=='Watershed based Segmentation':
			paintWaterComp(2)
		elif choice2=='Cluster based segmentation':
			paintClusterComp(2)

		compare(choice1,choice2)






if choice=='Image Pre Processing':
	painter(1)
elif choice=='Image Segmentation':
	painter(2)
elif choice=='Comparision':
	painter(3)





