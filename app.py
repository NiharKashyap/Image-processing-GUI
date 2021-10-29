import cv2
import streamlit as st
from PIL import Image
import selfieSeg as ss
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
	('Image Pre Processing', 'Image Segmentation'))


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
		d = st.sidebar.slider('Kernel Size', 0, 20, 6, help='We travel through the image with this filter by applying the desired operation.')
		sigmaColor = st.sidebar.slider('Sigma Color', 0, 100, 75)
		sigmaSpace = st.sidebar.slider('Sigma Space', 0, 100, 75)

		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.bilateral(frame, d, sigmaColor,sigmaSpace)
			paintImage()


	elif choice=='Non Local Means':
		level = st.sidebar.slider('Window size', 1, 10, 2, help='We travel through the image with this filter by applying the desired operation.')
		h = st.sidebar.slider('H', 1, 20, 2)
		hColor = st.sidebar.slider('HColor', 1, 20, 2)
		frame = preProcess(uploaded_file)
		ip.denoising(frame, level, h, hColor)
		paintImage()


def paintContrast():
	choice = st.sidebar.selectbox('Select Algorithm', ('Histogram Equilization', 
		'CLAHE', 'Min-Max Contrast Stretching'))

	if choice=='Histogram Equilization':
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.equi_hist(frame)
			paintImage()
		
	elif choice=='CLAHE':
		clipLim = st.sidebar.slider('Clip Size', 1,100,15)
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.clahe(frame, clipLim)
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
	
	choice = st.sidebar.selectbox('Select Algorithm', ('Global', 
		'Local', 'Adaptive', 'Otsu'))

	if choice=='Global':
		thresh = st.sidebar.slider('Threshold Value', 0, 100, 127, help='We travel through the image with this filter by applying the desired operation.')
		maxVal = st.sidebar.slider('Max Value', 0, 100, 127, help='We travel through the image with this filter by applying the desired operation.')
		
		frame = preProcess(uploaded_file)
		if frame is not None:
			ip.globalThresh(frame, thresh, maxVal)
			paintImage()


def paintEdge():
	st.title('Edge Detection')
	choice = st.sidebar.selectbox('Select Algorithm', ('Canny', 
		'Sobel', 'Prewitt', 'LoG', 'Roberts'))

def paintReg():
	st.title('Region based Segmentation')

def paintWater():
	st.title('Watershed based Segmentation')


def paintActive():
	st.title('Active Contour model based Segmentation')

def paintRandomWalk():
	st.title('Random walker segmentation')

def paintCluster():
	st.title('Cluster based segmentation')

def painter(id):
	if id==1:
		choice = st.sidebar.selectbox('Select PreProcessing Technique',
		('Denoising', 'Contrast Enhancement', 'Brightness Enhancement',
		'Sharpness Enhancement'))
	
	elif id==2:
		choice = st.sidebar.selectbox('Select Segmentation Technique',
		('Thresholding', 'Edge Detection', 'Region based Segmentation',
		'Watershed based Segmentation', 'Active Contour model based Segmentation',
		'Random walker segmentation', 'Cluster based segmentation'))

	if choice=='Denoising':
		paintDenoise()
	elif choice=='Contrast Enhancement':
		paintContrast()
	elif choice=='Brightness Enhancement':
		paintBrightness()
	elif choice=='Sharpness Enhancement':
		paintSharp()
	elif choice=='Thresholding':
		paintThresh()
	elif choice=='Edge Detection':
		paintEdge()
	elif choice=='Region based Segmentation':
		paintReg()
	elif choice=='Watershed based Segmentation':
		paintWater()
	elif choice=='Active Contour model based Segmentation':
		paintActive()
	elif choice=='Random walker segmentation':
		paintRandomWalk()
	elif choice=='Cluster based segmentation':
		paintCluster()






if choice=='Image Pre Processing':
	painter(1)
elif choice=='Image Segmentation':
	painter(2)




def ImgEnhance():
	uploaded_file = st.file_uploader("Choose a Image file")
	alpha = st.sidebar.slider('Contrast', 1.0, 3.0, 1.25)
	beta = st.sidebar.slider('Bleftness', 0, 100, 25)
	
	Oimage, Simage = st.columns(2)

	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		my_img = cv2.imread('tempDir/' + uploaded_file.name)
		#my_img = Image.open(uploaded_file)
		frame = np.array(my_img)
		frame = cv2.resize(frame, (640, 480))
		ip.bleftness_enhancement(frame, alpha, beta)
		
		imgO = Image.open('imgO.jpg')
		imgS = Image.open('imgS.jpg')
		Oimage.image(imgO, use_column_width=True)
		Oimage.header("Original")
		Simage.image(imgS, use_column_width=True)
		Simage.header("Processed")

def denoise():

	uploaded_file = st.file_uploader("Choose a Image file")
	level = st.sidebar.slider('Window size', 1, 10, 2)
	h = st.sidebar.slider('H', 1, 20, 2)
	hColor = st.sidebar.slider('HColor', 1, 20, 2)
	
	Oimage, Simage = st.columns(2)

	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		my_img = cv2.imread('tempDir/' + uploaded_file.name)
		frame = np.array(my_img)
		frame = cv2.resize(frame, (640, 480))

		ip.denoising(frame, level, h, hColor)
		
		imgO = Image.open('imgO.jpg')
		imgS = Image.open('imgS.jpg')
		Oimage.image(imgO, use_column_width=True)
		Oimage.header("Original")
		Simage.image(imgS, use_column_width=True)
		Simage.header("Processed")



