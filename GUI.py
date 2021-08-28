import cv2
import streamlit as st
from PIL import Image
import selfieSeg as ss
import numpy as np
import os

choice = st.sidebar.selectbox('Select Your Application', 
	('Image background Removal','Live background Removal','Background3'))

def Livback():
	background = st.sidebar.selectbox('Select Your background', 
		('Background1','Background2','Background3'))

	thresh = st.sidebar.slider('Threshold', 0.0, 1.0, 0.25)

	if background=='Background1':
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/1.jpg'
	elif background=='Background2':
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/2.jpg'
	else:
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/3.jpg'

	vid = cv2.VideoCapture(0)

	vid.set(3, 640)
	vid.set(4, 480)

	Oimage = st.empty()
	Simage = st.empty()

	while True:

		ret, frame = vid.read()
		ss.rem(frame, filepath,thresh)
		imgO = Image.open('imgO.jpg')
		imgS = Image.open('imgS.jpg')
		Oimage.image(imgO)
		Simage.image(imgS)


def Imgback():

	background = st.sidebar.selectbox('Select Your background', 
		('Background1','Background2','Background3'))

	thresh = st.sidebar.slider('Threshold', 0.0, 1.0, 0.25)

	if background=='Background1':
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/1.jpg'
	elif background=='Background2':
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/2.jpg'
	else:
		filepath = 'C:/Code/Projects/Image Processing GUI/BackgroundImages/3.jpg'


	uploaded_file = st.file_uploader("Choose a Image file")


	Oimage, Simage = st.columns(2)

	if uploaded_file is not None:
		with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
			f.write(uploaded_file.getbuffer())
		my_img = cv2.imread('tempDir/' + uploaded_file.name)
		#my_img = Image.open(uploaded_file)
		frame = np.array(my_img)
		frame = cv2.resize(frame, (640, 480))

		ss.rem(frame, filepath,thresh)
		imageLocation = st.empty()
		imgO = Image.open('imgO.jpg')
		imgS = Image.open('imgS.jpg')
		Oimage.image(imgO, use_column_width=True)
		Oimage.header("Original")
		Simage.image(imgS, use_column_width=True)
		Simage.header("Processed")





if choice=='Image background Removal':
	Imgback()
elif choice=='Live background Removal':
	Livback()