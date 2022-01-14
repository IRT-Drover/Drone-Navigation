# Converting pixels of the path into coordinates

import cv2 as cv
import numpy as np
import glob
from astar import Node

def main():
    PATH = []
    # newNode = Node()
    # print(newNode)
    img = cv.imread(glob.glob('testmap.png'))

    height_pix, width_pix = img.shape[:2]
    resolution =  72 #pixels per inch
    height = height_pix/resolution
    width = width_pix/resolution

    center_pix = Node()
    center_pix.y = int((len(img))/2-1)
    center_pix.x = int((len(img[0]))/2-1)
    y_center = center_pix.y
    x_center = center_pix.x

    lat_center = 51.606733
    long_center = -3.986813
    
    conversion = 111111

    y_b = 399
    y_a = 781
    magnification = ((y_center*(y_b-y_a) + 2*lat_center*(y_a-y_b)) * resolution) / (y_b*conversion*(y_a-y_center)-y_a*conversion*(y_b-y_center))

    waypoints = []

    for pixel in PATH:
        y = pixel.y
        x = pixel.x
        
        dist_y = (y-y_center)/resolution
        dist_x = (x-x_center)/resolution

        lat = dist_y * magnification * conversion + lat_center
        long = dist_x * magnification * conversion + long_center

        waypoints.append([lat,long])

    
