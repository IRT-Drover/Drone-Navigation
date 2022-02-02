import numpy as np
import cv2 as cv

# Initiating Camera
camera = cv.VideoCapture(1)

# Check frame dimensions
print("Checking Camera Dimensions...")
(grabbed, frame) = camera.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
if grabbed == False:
    raise Exception('Unsuccessful')
else:
    print(f'Successful: {fwidth}(H) {fheight}(V)')
    
 
# Define the codec and create VideoWriter object
# Windows
# fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('CameraCalibrationVideo.avi', fourcc, 20.0, (fwidth,fheight))

# Mac
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('CameraCalibrationVideo1.mp4', fourcc, 20.0, (fwidth,fheight))
print("Video started...")
while camera.isOpened():
    ret, frame = camera.read()