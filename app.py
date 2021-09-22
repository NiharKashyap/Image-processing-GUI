import cv2
import streamlit as st
from PIL import Image
import selfieSeg as ss
import imageProcessing as ip
import numpy as np
import os

st.set_page_config(layout='wide')
left,center,right = st.columns([3,8,2])


choice = left.selectbox('Select Your Application', 
	('Image Pre Processing', 'Image Segmentation'))


uploaded_file = center.file_uploader("Choose a Image file")

def paintDenoise():
	center.title('Denoise Image')
	choice = right.selectbox('Select Algorithm', ('Image Blur', 'Gaussian Filter',
		'Median Filter', 'Laplacian Filter', 'Non Local Means'))

	if choice=='Image Blur':
		thresh = right.slider('Kernel Size', 0, 20, 6)

	elif choice=='Gaussian Filter':
		thresh = right.slider('Kernel Size', 0, 20, 6)
	elif choice=='Median Filter':
		thresh = right.slider('Kernel Size', 0, 20, 6)
	elif choice=='Laplacian Filter':
		thresh = right.slider('Kernel Size', 0, 20, 6)
	elif choice=='Non Local Means':
		level = right.slider('Window size', 1, 10, 2)
		h = right.slider('H', 1, 20, 2)
		hColor = right.slider('HColor', 1, 20, 2)

		Oimage = center.empty()
		Simage = center.empty()

		if uploaded_file is not None:
			with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
				f.write(uploaded_file.getbuffer())
			my_img = cv2.imread('tempDir/' + uploaded_file.name)
			frame = np.array(my_img)
			frame = cv2.resize(frame, (640, 480))

			ip.denoising(frame, level, h, hColor)
			
			imgO = Image.open('imgO.jpg')
			imgS = Image.open('imgS.jpg')
			Oimage.image(imgO)
			#Oimage.header("Original")
			Simage.image(imgS)
			#Simage.header("Processed")


def paintContrast():
	center.title('Contrast Enhancement')
	choice = right.selectbox('Select Algorithm', ('Histogram Equilization', 
		'CLAHE', 'Min-Max Contrast Stretching'))

def paintBright():
	center.title('Brightness Enhancement')


def paintSharp():
	center.title('Sharpness Enhancement')

def paintThresh():
	center.title('Thresholding')
	choice = right.selectbox('Select Algorithm', ('Global', 
		'Local', 'Adaptive', 'Otsu'))

def paintEdge():
	center.title('Edge Detection')
	choice = right.selectbox('Select Algorithm', ('Canny', 
		'Sobel', 'Prewitt', 'LoG', 'Roberts'))

def paintReg():
	center.title('Region based Segmentation')

def paintWater():
	center.title('Watershed based Segmentation')


def paintActive():
	center.title('Active Contour model based Segmentation')

def paintRandomWalk():
	center.title('Random walker segmentation')

def paintCluster():
	center.title('Cluster based segmentation')

def painter(id):
	if id==1:
		choice = right.selectbox('Select PreProcessing Technique',
		('Denoising', 'Contrast Enhancement', 'Brightness Enhancement',
		'Sharpness Enhancement'))
	
	elif id==2:
		choice = right.selectbox('Select Segmentation Technique',
		('Thresholding', 'Edge Detection', 'Region based Segmentation',
		'Watershed based Segmentation', 'Active Contour model based Segmentation',
		'Random walker segmentation', 'Cluster based segmentation'))

	if choice=='Denoising':
		paintDenoise()
	elif choice=='Contrast Enhancement':
		paintContrast()
	elif choice=='Brightness Enhancement':
		paintBright()
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
	beta = st.sidebar.slider('Brightness', 0, 100, 25)
	
	Oimage, Simage = st.columns(2)

	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		my_img = cv2.imread('tempDir/' + uploaded_file.name)
		#my_img = Image.open(uploaded_file)
		frame = np.array(my_img)
		frame = cv2.resize(frame, (640, 480))
		ip.brightness_enhancement(frame, alpha, beta)
		
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



