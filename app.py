#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''