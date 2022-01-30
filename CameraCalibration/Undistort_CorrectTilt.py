#Undisort.py
#Created by Chris Rillahan
#Last Updated: 02/04/2015
#Written with Python 2.7.2, OpenCV 2.4.8 and NumPy 1.8.0

#This program takes a video file and removes the camera distortion based on the
#camera calibration parameters.  The filename and the calibration data filenames
#should be changed where appropriate.  Currently, the program is set to search for
#these files in the folder in which this file is located.

#This program first loads the calibration data.  Secondly, the video is loaded and
#the metadata is derived from the file.  The export parameters and file structure
#are then set-up.  The file then loops through each frame from the input video,
#undistorts the frame and then saves the resulting frame into the output video.
#It should be noted that the audio from the input file is not transfered to the
#output file.

import numpy as np
import cv2, time, sys
import glob

# filename = 'GOPR0028.MP4'
directory = 'pixeltocoordinate_imagetesting/'

print('Loading data files')

npz_calib_file = np.load('calibration_data.npz')

# print('total reprojection error: ' + npz_calib_file['reproj_error'])

distCoeff = npz_calib_file['distCoeff']
intrinsic_matrix = npz_calib_file['intrinsic_matrix']

npz_calib_file.close()

print('Finished loading files')
print(' ')
print('Starting to undistort the video....')

#Loading images
IMAGES = glob.glob(f'{directory}*')
for i in range(0, len(IMAGES)):
    print ('Loading... Calibration Image... ' + IMAGES[i][len(directory):])
    image = cv2.imread(IMAGES[i])

    # undistort
    undst = cv2.undistort(image, intrinsic_matrix, distCoeff, None)

    cv2.imshow('Undisorted Image',undst)
    cv2.imwrite(f'{directory}undst-{IMAGES[i][len(directory):]}', undst)