# Converting pixels of the path into coordinates

import cv2 as cv
import numpy as np
import glob
import math
from astar import Node

# Geodesics #May have to change in the future if scale really big. because arc distance ≠ distance calculated using proportions and image points bc it's flat
# Methods for converting GPS points to distance: https://blog.mapbox.com/fast-geodesic-approximations-with-cheap-ruler-106f229ad016
# Method with ~0% margin of error for calculating distance given the coordinates at a scale smaller than 100 mi:
# https://www.govinfo.gov/content/pkg/CFR-2005-title47-vol4/pdf/CFR-2005-title47-vol4-sec73-208.pdf
# Calculates meters per degree of longitude and latitude based on latitude using above method: https://github.com/mapbox/cheap-ruler-cpp/blob/master/include/mapbox/cheap_ruler.hpp
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
    #Camera Specs
    focal = 0.002 # focal length in meters #x=sensor_hdimens*pixelsize_in_micrometers/1,000,000 #f = x/(2tan(angleofview/2))
    unitcell = 1.12 # size of single pixel in micrometers
    resolution =  1/unitcell*1000000 # pixels per meter
    sensor_H = 4656 # pixel dimensions of camera sensor/photo
    sensor_V = 3496
    
    
    PATH = []
    for i in [[0,0], [300,300]]:
        testpixel = Node()
        testpixel.x = i[0]
        testpixel.y = i[1]
        PATH.append(testpixel)
    print(PATH)
    
    # newNode = Node()
    # print(newNode)
    img = cv.imread('Maze_1.png')
    
    
    # height_pix, width_pix = img.shape[:2]
    # print(height_pix)
    # print(width_pix)
    
    # imageheight = height_pix/resolution
    # imagewidth = width_pix/resolution
    
    altitude = 10 # in meters
    image_dist = (altitude * focal) / (altitude + focal) # not sure if physics
    # mapheight = imageheight * altitude / image_dist # not sure if physics is right
    magnification = image_dist/altitude #imageheight / mapheight #not sure if physics right
    print("image distance: " + str(image_dist))
    print("altitude: " + str(altitude))
    print("magnification: " + str(magnification))
    

    # image_center = Node()
    # pixel_x_0 = int((len(img[0]))/2-1) # pixel x-coordinate 
    # pixel_y_0 = int((len(img))/2-1) # pixel y-coordinate
    pixel_x_0 = int((sensor_H)/2-1) # pixel x-coordinate 
    pixel_y_0 = int((sensor_V)/2-1) # pixel y-coordinate
    
    long_0 = -74.56913024979528 #longitude
    lat_0 = 40.61865173982036 #latitude
    

    MPD_long = MPD(lat_0)[0]
    MPD_lat = MPD(lat_0)[1]

    waypoints = []
    
    for pixel in PATH:
        distance_img_x = (pixel_x_0 - pixel.x) / resolution
        distance_map_x = distance_img_x / magnification
        print("horiz meters to center: " + str(distance_map_x))
        long_1 = long_0 + distance_map_x / MPD_long
        
        distance_img_y = (pixel_y_0 - pixel.y) / resolution
        distance_map_y = distance_img_y / magnification
        print("vert meters to center: " + str(distance_map_y))
        lat_1 = lat_0 + distance_map_y / MPD_lat
        
        waypoints.append([lat_1, long_1])
    
    print(MPD_long)
    print(MPD_lat)
    print([lat_0,long_0])
    print(waypoints)
    return waypoints


waypoint = main()

lat, long = waypoint[0]



# f = 500
# rotXval = 90
# rotYval = 90
# rotZval = 90
# distXval = 500
# distYval = 500
# distZval = 500

# def onFchange(val):
#     global f
#     f = val
# def onRotXChange(val):
#     global rotXval
#     rotXval = val
# def onRotYChange(val):
#     global rotYval
#     rotYval = val
# def onRotZChange(val):
#     global rotZval
#     rotZval = val
# def onDistXChange(val):
#     global distXval
#     distXval = val
# def onDistYChange(val):
#     global distYval
#     distYval = val
# def onDistZChange(val):
#     global distZval
#     distZval = val

# if __name__ == '__main__':

#     #Read input image, and create output image
#     src = cv.imread('CalibratingCamera/CheckerboardPhotos_1/checkerboard0.jpg')
#     src = cv.resize(src,(640,480))
#     dst = np.zeros_like(src)
#     h, w = src.shape[:2]

#     #Create user interface with trackbars that will allow to modify the parameters of the transformation
#     wndname1 = "Source:"
#     wndname2 = "WarpPerspective: "
#     cv.namedWindow(wndname1, 1)
#     cv.namedWindow(wndname2, 1)
#     cv.createTrackbar("f", wndname2, f, 1000, onFchange)
#     cv.createTrackbar("Rotation X", wndname2, rotXval, 180, onRotXChange)
#     cv.createTrackbar("Rotation Y", wndname2, rotYval, 180, onRotYChange)
#     cv.createTrackbar("Rotation Z", wndname2, rotZval, 180, onRotZChange)
#     cv.createTrackbar("Distance X", wndname2, distXval, 1000, onDistXChange)
#     cv.createTrackbar("Distance Y", wndname2, distYval, 1000, onDistYChange)
#     cv.createTrackbar("Distance Z", wndname2, distZval, 1000, onDistZChange)

#     #Show original image
#     cv.imshow(wndname1, src)

#     k = -1
#     while k != 27:

#         if f <= 0: f = 1
#         rotX = (rotXval - 90)*np.pi/180
#         rotY = (rotYval - 90)*np.pi/180
#         rotZ = (rotZval - 90)*np.pi/180
#         distX = distXval - 500
#         distY = distYval - 500
#         distZ = distZval - 500

#         # Camera intrinsic matrix
#         K = np.array([[f, 0, w/2, 0],
#                     [0, f, h/2, 0],
#                     [0, 0,   1, 0]])

#         # K inverse
#         Kinv = np.zeros((4,3))
#         Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
#         Kinv[-1,:] = [0, 0, 1]

#         # Rotation matrices around the X,Y,Z axis
#         RX = np.array([[1,           0,            0, 0],
#                     [0,np.cos(rotX),-np.sin(rotX), 0],
#                     [0,np.sin(rotX),np.cos(rotX) , 0],
#                     [0,           0,            0, 1]])

#         RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
#                     [            0, 1,            0, 0],
#                     [ -np.sin(rotY), 0, np.cos(rotY), 0],
#                     [            0, 0,            0, 1]])

#         RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
#                     [ np.sin(rotZ), np.cos(rotZ), 0, 0],
#                     [            0,            0, 1, 0],
#                     [            0,            0, 0, 1]])

#         # Composed rotation matrix with (RX,RY,RZ)
#         R = np.linalg.multi_dot([ RX , RY , RZ ])

#         # Translation matrix
#         T = np.array([[1,0,0,distX],
#                     [0,1,0,distY],
#                     [0,0,1,distZ],
#                     [0,0,0,1]])

#         # Overall homography matrix
#         H = np.linalg.multi_dot([K, R, T, Kinv])

#         # Apply matrix transformation
#         cv.warpPerspective(src, H, (w, h), dst, cv.INTER_NEAREST, cv.BORDER_CONSTANT, 0)

#         # Show the image
#         cv.imshow(wndname2, dst)
#         k = cv.waitKey(1)