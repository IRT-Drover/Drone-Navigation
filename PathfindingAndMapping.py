import cv2 as cv
import numpy as np
import simplekml
import os
import glob
import datetime

from astar import aStar
from PixelstoCoordinates import pixelstocoordinates

# PathfindingAndMapping connects all pathfinding and mapping programs in one method without losing ability to test each
# method independently.

# On one given image, astar() takes in the image name and image number and calls the aStar algorithm from astar.py.
# The algorithm calculates the pixel path, makes a copy of the image and draws the path on the copied image, and returns the pixel path as a list.
# astar() appends the pixel path to astarData.txt (after creating the txt) and returns the pixel path.
# On one given image, pixelToCoords() takes in a pixel path and image number and calls the pixel to coordinates algorithm from PixelstoCoordinates.py.
# The algorithm takes in the pixel path and image data, calculates the GPS coordinates of the pixel path, and returns the GPS path
# pixelToCoords() appends the GPS path to coordsData.txt and saves/updates the path in the numpy file GPSDATA_TOSEND.npz (Note: saved within another list, so it's a 2d list)

# pathfindingandmapping() puts the previous methods together to evaluate all photos in a flight.
# The method stores the GPS paths of every photo in a 2D list, which is then saved in numpy file GPSDATA_TOSEND.npz.
# Order of paths in the 2D list correspond to order of images taken from oldest to newest
# (i.e. first path corresponds with first image (oldest image) and last path corresponds with the last image (newest image))
# The navigation script will create its waypoints by accessing GPSDATA_TOSEND.npz, which will then be sent to the rover

# HOW TO USE: Create a PathfindingAndMapping object. Call the pathfindingandmapping method
# GPS paths of all images in a flight are stored in GPSDATA_TOSEND as a 2D list. Each object of the 2D list is a GPS path for one image
class PathfindingAndMapping:
    def __init__(self, directory_flights):
        self.setflight_recent(directory_flights)

    def getflight(self):
        return self.flight

    def setflight(self, flight):
        self.flight = flight

    # Sets flight directory to the most recent flight
    def setflight_recent(self, directory_flights):
        flights = glob.glob(directory_flights+"*")
        self.flight = max(flights, key=os.path.getmtime) + '/'

    # Parameters: name of image file, image number (starting from 1)
    # Assumes astar data file is 'astarData.txt'
    # Runs astar algorithm on the image. Default start is top left and default end is bottom right of image
    # Astar algorithm returns pixel path as a list and saves image with path drawn on it
    # Appends path to astarData.txt and returns path
    def astar(self, image_name, img_num, start=-1, end=-1):
        # Calculates path, draws path on image and saves image to flight, returns path
        pixel_PATH, img = aStar(image_name, start, end, self.flight)
        # Appends data to 'astarData.txt'
        self.writeFile("astarData.txt", pixel_PATH, img_num)
        return pixel_PATH

    # Parameters: pixel path, image number (starting from 1)
    # Assumes picture data file is 'pictureData.txt'
    # Saves home coordinate as a list
    # Saves picture data for an image as list --> [coordinate, altitude]
    # Runs pixels to coordinates algorithm, which takes pixel path and picture data as inputs
    # Appends GPS path to coordsData.txt
    # Saves GPS path to numpy file, which will be sent to rover, within a 2D list.
    # Returns the GPS path as a list
    def pixelToCoords(self, pixel_PATH, img_num):
        fo = open(self.flight + 'pictureData.txt')
        pictureData = fo.readlines()
        fo.close()
        imgdata_index = (img_num - 1) * 3 + 1
        selectpicdata = pictureData[imgdata_index:imgdata_index+2]
        
        # Stores GPS coordinates of drone as a list as the first element in selectpicdata.
        droneLoc = [float(selectpicdata[0][:selectpicdata[0].find(' ')]), float(selectpicdata[0][selectpicdata[0].find(' ') + 1:])]
        selectpicdata[0] = droneLoc
        # Stores height as the second element in selectpicdata
        selectpicdata[1] = float(selectpicdata[1])
        
        # Calculates and returns GPS path
        GPS_PATH = pixelstocoordinates(pixel_PATH, selectpicdata)
        
        # Appends data to 'coordsData.txt'
        self.writeFile("coordsData.txt", GPS_PATH, img_num)
        
        # Saves data to npz file
        np.savez(self.flight+'GPSDATA_TOSEND', GPSPATHS=[GPS_PATH])
        
        return GPS_PATH

    # Parameter: 3D list of start pixel and end pixel for each image # May get rid of later
    # Collects all images in a list and sorts them from oldest to newest.
    # Runs astar and pixel to coordinates algorithm.
    # Creates text file 'astarData.txt'for astar output and 'coordsData.txt' for pixeltocoord output.
    # Appends pixel path results to astarData.txt and saves image with path drawn on it.
    # Appends coordinate path results to coordsData.txt
    # Saves GPS path of every image to numpy file, which will be sent to rover, within a 2D list. Each element is a path for an image
    def pathfindingandmapping(self, startend_list):
        images = glob.glob(self.flight + "*[!-p].png")
        images.sort(key=os.path.getmtime)
        GPSPaths = []
        for img_num in range(1, len(images) + 1):
            pixelPath = self.astar(images[img_num-1][len(self.flight):], img_num+2, startend_list[img_num-1][0], startend_list[img_num-1][1])
            GPSPaths.append(self.pixelToCoords(pixelPath, img_num+2))
            
        # Saves data as a 2D list to npz file
        np.savez(self.flight+'GPSDATA_TOSEND', GPSPATHS=GPSPaths)
        return GPSPaths

    # Creates text file called fn in flight folder if not already created
    # Appends 'Picture #' and data in separate lines
    def writeFile(self, fn, data, pic_num):
        fo = open(self.flight + fn, "a")
        # Append data, after converted to string, to the end of file
        fo.write("Picture " + str(pic_num) + '\n')
        fo.write(self.twoD_ListToString(data) + '\n')

        fo.close()
    
    # Converts 2D list of coordinates to a string: x,y x,y x,y
    def twoD_ListToString(self, list2d):
        string_list = ''
        for i in range(len(list2d)):
            string_list += ' ' + str(list2d[i][0])
            for j in range(1, len(list2d[i])):
                string_list += ',' + str(list2d[i][j])
        string_list = string_list[1:]
        
        return string_list
            
            
# Testing pathfindingandmapping method
flightpathmap = PathfindingAndMapping('DronePictures/')
print(flightpathmap.getflight())

# Football field
# startendlist = [[[720,900],[720,1]]] #check if pixel selection from opencv selects index or pixel number
# Park
# startendlist = [[[1300,1],[10,686]]]
# Trail
startendlist = [[[1170,1],[1,900]]]
GPSPaths = flightpathmap.pathfindingandmapping(startendlist)
# gpsdata = np.load('DronePictures/2021-04-19 18:08:11.670318/GPSDATA_TOSEND.npz')
# for i in gpsdata['GPSPATHS']:
#     for j in i:
#         print(j)

# Creating kmz file to view on maps
npzfile = np.load(flightpathmap.getflight()+'GPSDATA_TOSEND.npz')
GPSPaths = npzfile['GPSPATHS']
kml=simplekml.Kml()
for coord in GPSPaths[0]:
    kml.newpoint(coords=[(coord[1],coord[0])])
kml.save(flightpathmap.getflight()+'trail-p.kml')