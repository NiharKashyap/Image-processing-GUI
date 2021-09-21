import cv2
import streamlit as st
from PIL import Image
import selfieSeg as ss
import imageProcessing as ip
import numpy as np
import os

choice = st.sidebar.selectbox('Select Your Application', 
	('Image Pre Processing', 'Image Segmentation'))

def paintDenoise():
	st.title('Denoise Image')
	choice = st.selectbox('Select Algorithm', ('Image Blur', 'Gaussian Filter',
		'Median Filter', 'Laplacian Filter', 'Non Local Means'))

def paintContrast():
	st.title('Contrast Enhancement')
	choice = st.selectbox('Select Algorithm', ('Histogram Equilization', 
		'CLAHE', 'Min-Max Contrast Stretching'))

def paintBright():
	st.title('Brightness Enhancement')


def paintSharp():
	st.title('Sharpness Enhancement')



def painter(id):
	if id==1:
		choice = st.sidebar.selectbox('Select PreProcessing Technique',
		('Denoising', 'Contrast Enhancement', 'Brightness Enhancement',
		'Sharpness Enhancement'))
	
	elif id==2:
		choice = st.sidebar.selectbox('Select Segmentation Technique',
		('Thresholding' 'Edge Detection', 'Region based Segmentation',
		'Watershed based Segmentation', 'Active Contour model based Segmentation',
		'Random walker segmentation', 'Clsuter based segmentation'))

	if choice=='Denoising':
		paintDenoise()
	elif choice=='Contrast Enhancement':
		paintContrast()
	elif choice=='Brightness Enhancement':
		paintBright()
	elif choice=='Sharpness Enhancement':
		paintSharp()



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



