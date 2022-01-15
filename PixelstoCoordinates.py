# Converting pixels of the path into coordinates

import cv2 as cv
import numpy as np
import glob
import math
from astar import Node

# Geodesics
# Methods for converting GPS points to distance: https://blog.mapbox.com/fast-geodesic-approximations-with-cheap-ruler-106f229ad016
# Method with ~0% margin of error for calculating distance given the coordinates at a scale smaller than 100 mi:
# https://www.govinfo.gov/content/pkg/CFR-2005-title47-vol4/pdf/CFR-2005-title47-vol4-sec73-208.pdf
# Calculating coordinates based on distance using above method (Based on the destination method): https://github.com/mapbox/cheap-ruler-cpp/blob/master/include/mapbox/cheap_ruler.hpp
# WGS84 ellipsoid model of the Earth: https://ahrs.readthedocs.io/en/latest/wgs84.html

def MPD(lat_0):
    equator_radius = 6378137.0
    flattening_factor = 1/298.257223563
    E2 = flattening_factor * (2 - flattening_factor)
    radians = math.pi/180


    mul = math.pi/180 * equator_radius * 1000 # meters per 1 degree longitude at equator
    w2 = 1 / (1 - E2 * (1 - (math.cos(lat_0*radians)) ** 2)) # I think w2 and w are values that change mul based on latitude
    w = math.sqrt(w2)

    MPD_long = mul * w * math.cos(lat_0 * radians) # meters per degree longitude
    MPD_lat = mul * w * w2 * (1 - E2) # meters per degree latitude
    
    return [MPD_long, MPD_lat]

def main():
    testpixel = Node()
    testpixel.x = 50
    testpixel.y = 100

    PATH = [testpixel]
    # newNode = Node()
    # print(newNode)
    img = cv.imread('Maze_1.png')
    
    altitude = 5 # in meters
    focal = 0.003 # in meters
    image_dist = (altitude * focal) / (altitude + focal) # may not be true depending on type of camera
    magnification = image_dist / altitude # may not be true depending on type of camera
    resolution =  3000 #pixels per meter #~72 ppi

    # height_pix, width_pix = img.shape[:2]
    
    # imageheight = height_pix/resolution
    # imagewidth = width_pix/resolution
    

    # image_center = Node()
    pixel_x_0 = int((len(img[0]))/2-1) # pixel x-coordinate 
    pixel_y_0 = int((len(img))/2-1) # pixel y-coordinate
    
    long_0 = -74.56913024979528 #longitude
    lat_0 = 40.61865173982036 #latitude
    

    MPD_long = MPD(lat_0)[0]
    MPD_lat = MPD(lat_0)[1]

    waypoints = []
    
    for pixel in PATH:
        distance_img_x = (pixel_x_0 - pixel.x) / resolution
        distance_map_x = distance_img_x / magnification
        long_1 = long_0 + distance_map_x / MPD_long
        
        distance_img_y = (pixel_y_0 - pixel.y) / resolution
        distance_map_y = distance_img_y / magnification
        lat_1 = lat_0 + distance_map_y / MPD_lat
        
        waypoints.append([lat_1, long_1])
    
    print(MPD_long)
    print(MPD_lat)
    print(waypoints)
    return waypoints


main()

lat_0 = 53.32055555555556



