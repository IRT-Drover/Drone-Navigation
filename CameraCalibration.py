# Basic info on camera calibration: https://www.youtube.com/watch?v=x6YIwoQBBxA
# Using opencv to do camera calibration. https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# Estimating the extrinsic and, most importantly, the intrinsic (e.g. lens-distortion) parameters of the camera

# Import required modules
import cv2
import numpy as np
import os
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9) # based on lines, not on boxes
SQUARE_SIZE = 30

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) #check difference
# objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) #check difference
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0]*SQUARE_SIZE:SQUARE_SIZE,
							0:CHECKERBOARD[1]*SQUARE_SIZE:SQUARE_SIZE].T.reshape(-1, 2)

prev_img_shape = None

print(objectp3d)


# Extracting path of individual image stored
# in a given directory. it will take CheckerboardPhotos directory
# jpg files alone
images = glob.glob('CheckerboardPhotos/*.jpg')

for filename in images:
    image = cv2.imread(filename)
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	# If desired number of corners are
	# found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
					grayColor, CHECKERBOARD,
					cv2.CALIB_CB_ADAPTIVE_THRESH
					+ cv2.CALIB_CB_FAST_CHECK +
					cv2.CALIB_CB_NORMALIZE_IMAGE)

	# If desired number of corners can be detected then,
	# refine the pixel coordinates and display
	# them on the images of checker board
    if ret == True:
        threedpoints.append(objectp3d)

		# Refining pixel coordinates
		# for given 2d points.
        corners2 = cv2.cornerSubPix(
			grayColor, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

		# Draw and display the corners
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        cv2.imshow('img.png', image)
        print(threedpoints)
        cv2.waitKey(5000)


cv2.destroyAllWindows()

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	threedpoints, twodpoints,
    cv2.imread(images[0]).shape[:2], None, None) #gray.shape[::-1]


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

# UNDISTORTION

# Refines camera matrix.
# If alpha(4th parameter)=0, returns a matrix
# with minimum unwanted pixels, sometimes removing some pixels at image corners
# depending on time of distortion. If alpha=1, returns a matrix
# with all pixels retained with some extra black pixels.
# Also returns region of interest of photo (find out what roi means)
image = cv2.imread('CheckerboardPhotos/checkerboard1.jpg')
h, w = image.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h)) #figure out if we need this

# undistort
image_undistorted = cv2.undistort(image, matrix, distortion, None, newcameramatrix)
# crop the image
x, y, w, h = roi
image_undistorted = image_undistorted[y:y+h, x:x+w]
cv2.imwrite('CheckerboardPhotos/undistortedresult.png', image_undistorted)

# REPROJECTION ERROR. closer reprojection error is to zero,
# the more accurate the found parameters are
mean_error = 0
for i in range(len(threedpoints)):
    twodpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(twodpoints[i], twodpoints2, cv2.NORM_L2)/len(twodpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(threedpoints)))
