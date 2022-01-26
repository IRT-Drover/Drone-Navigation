# Basic info on camera calibration: https://www.youtube.com/watch?v=x6YIwoQBBxA
# Using opencv to do camera calibration. https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# Estimating the extrinsic and, most importantly, the intrinsic (e.g. lens-distortion) parameters of the camera

# Import required modules
import cv2
import numpy as np
import os
import glob

def main():
    global threedpoints, twodpoints, criteria
    # Define the dimensions of checkerboard
    CHECKERBOARD = (7, 7) # corners of internal
    SQUARE_SIZE = 30

    # Location of checkerboard images
    checkerboard_directory = 'CheckerboardPhotos_Multi_3/'

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
    			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    # 3D points real world coordinates of checkerboard
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0]*SQUARE_SIZE:SQUARE_SIZE,
    							0:CHECKERBOARD[1]*SQUARE_SIZE:SQUARE_SIZE].T.reshape(-1, 2)
    # objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
    # 							0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored
    # in a given directory. it will take CheckerboardPhotos directory
    # jpg files alone
    IMAGES = glob.glob(f'{checkerboard_directory}IMG_*.JPG') #MAKE THIS MORE EFFICIENT. FIGURE OUT BEST PLACE TO PUT THIS
    if len(IMAGES) == 0:
        raise Exception("no images found")
    print(IMAGES)

    # finds the corners of the checkerboard in the 2D image
    image_size = drawCorners(CHECKERBOARD, IMAGES, objectp3d, checkerboard_directory)

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    	threedpoints, twodpoints,
        image_size, None, None) #gray.shape[::-1]

    # Displaying required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)

    optimalcameramatrix = undistortion(IMAGES, matrix, distortion, checkerboard_directory)
    reprojectionError(threedpoints, twodpoints, r_vecs, t_vecs, optimalcameramatrix, distortion)

def drawCorners(CHECKERBOARD, images, objectp3d, checkerboard_directory):
    global threedpoints, twodpoints

    for filename in images:
        print("Loading..." + filename[-12:-4] + ".JPG")
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('grayscale', grayColor)
        # cv2.waitKey(4000)

    	# Finds the internal checkerboard board corners.
    	# If desired number of corners are
    	# found in the image then ret = true.
        # Corners is a list of pixel coordinates: row by row, left to right
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
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    image_size = cv2.imread(images[0]).shape[-2:-4:-1]
    #image_size = cv2.imread(images[0]).shape[:2]
    print(image_size)
    return image_size

def undistortion(images, matrix, distortion, checkerboard_directory):
    # Refines camera matrix.
    # If alpha(4th parameter)=0, returns a matrix
    # with minimum unwanted pixels, sometimes removing some pixels at image corners
    # depending on time of distortion. If alpha=1, returns a matrix
    # with all pixels retained with some extra black pixels.
    # Also returns region of interest of photo (find out what roi means)
    image = cv2.imread(images[0])
    h, w = image.shape[:2]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h)) # try changing alpha

    for file in images:
        image = cv2.imread(file)
        # undistort
        image_undistorted = cv2.undistort(image, matrix, distortion, None, newcameramatrix) #1
        # image_undistorted = cv2.undistort(image, matrix, distortion, None, matrix) #2
        # crop the image
        x, y, w, h = roi
        print(w)
        print(h)
        image_undistorted = image_undistorted[y:y+h, x:x+w]
        cv2.imwrite(f'{checkerboard_directory}undistortedresult{file[-12:-4]}.png', image_undistorted) #1
        # cv2.imwrite(f'{checkerboard_directory}/undistortedresult{filenumber+99}.png', image_undistorted) #2

    return newcameramatrix

# REPROJECTION ERROR. closer reprojection error is to zero,
# the more accurate the found parameters are
# basically mapping 3D points onto a 2D image and comparing it with our original image
def reprojectionError(threedpoints, twodpoints, r_vecs, t_vecs, matrix, distortion):
    mean_error = 0
    for i in range(len(threedpoints)):
        twodpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
        error = cv2.norm(twodpoints[i], twodpoints2, cv2.NORM_L2)/len(twodpoints2)
        mean_error += error
    print(f"total error: {mean_error/len(threedpoints)}")

main()
