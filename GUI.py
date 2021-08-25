import cv2
import streamlit as st
from PIL import Image
import selfieSeg as ss
import numpy as np

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

	Oimage = st.empty()
	Simage = st.empty()

	if uploaded_file is not None:

	  	my_img = Image.open(uploaded_file)
	  	frame = np.array(my_img)
	  	print(uploaded_file.name)
	  	orig=cv2.imread(uploaded_file.read())
	  	cv2.imwrite('orig.jpg', orig)
	  	frame = cv2.resize(frame, (640, 480))

	  	ss.rem(frame, filepath,thresh)
	  	imageLocation = st.empty()
	  	imgO = Image.open('imgO.jpg')
	  	imgS = Image.open('imgS.jpg')
	  	Oimage.image(imgO)
	  	Simage.image(imgS)
	




if choice=='Image background Removal':
	Imgback()
elif choice=='Live background Removal':
	Livback()