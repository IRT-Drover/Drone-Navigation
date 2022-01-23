# Converting pixels of the path into coordinates v2

import cv2 as cv
import numpy as np
import glob
import math
from astar import Node

# PyGeodesy Github: https://github.com/mrJean1/PyGeodesy
# Documentation: https://mrjean1.github.io/PyGeodesy/
from pygeodesy.ellipsoidalVincenty import LatLon
from pygeodesy import Datums

def main():
    #Camera Specs
    focal = 0.002 # focal length in meters #x=sensor_hdimens*pixelsize_in_micrometers/1,000,000 #f = x/(2tan(angleofview/2))
    unitcell = 1.12 # size of single pixel in micrometers
    resolution = 1/unitcell*1000000 # pixels per meter
    sensor_H = 4656 # pixel dimensions of camera sensor/photo
    sensor_V = 3496
    
    
    PATH = []
    for i in [[0,0],[4656/2-1,0],[4656-1,0],[4656-1,3496/2-1],[4656-1,3496-1],[4656/2-1,3496-1],[0,3496-1], [0,3496/2-1]]:
        testpixel = Node()
        testpixel.x = i[0]
        testpixel.y = i[1]
        PATH.append(testpixel)
    
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
    # pixel_y_0 = int((len(img))/2-1) # pixel y-coordinate
    # pixel_x_0 = int((len(img[0]))/2-1) # pixel x-coordinate 
    pixel_y_0 = int((sensor_V)/2-1) # pixel y-coordinate
    pixel_x_0 = int((sensor_H)/2-1) # pixel x-coordinate 
    
    lat_0 = 40.61865173982036 #latitude
    long_0 = -74.56913024979528 #longitude
    drone = LatLon(40.61865173982036, -74.56913024979528, datum=Datums.NAD83) # default datum is WGS-84

    # MPD_long = MPD(lat_0)[0]
    # MPD_lat = MPD(lat_0)[1]

    rover_path = []
    
    print("---")
    
    testingpath=[]
    for pixel in PATH:
        distance_img_y = (pixel_y_0 - pixel.y) / resolution
        distance_map_y = distance_img_y / magnification
        print("Vert meters to center: " + str(distance_map_y))
        
        distance_img_x = (pixel.x - pixel_x_0) / resolution
        distance_map_x = distance_img_x / magnification
        print("Horiz meters to center: " + str(distance_map_x))
        
        
        bearing = 0 # compass 360 degrees; north = 0 degrees
        if distance_map_x != 0:
            bearing = 90 - (math.atan(distance_map_y/distance_map_x)*180/math.pi)
            if distance_map_x < 0:
                bearing += 180
        elif distance_map_y < 0:
            bearing = 180

        distance_map = math.sqrt(distance_map_x**2 + distance_map_y**2)
        
        print("\nBearing: " + str(bearing))
        print("Distance: " + str(distance_map) + "\n")
        # http://www.movable-type.co.uk/scripts/latlong-vincenty.html
        # Millimeter accuracy. To get to nanometer accuracy, will have to switch from Vincenty to Karney's method
        waypoint = drone.destination(distance_map, bearing)
        
        rover_path.append([waypoint.lat, waypoint.lon])
        testingpath.append(waypoint)
    
    print("Drone Coordinate: ")
    print(drone)
    print([lat_0,long_0])
    print("Path:")
    for i in testingpath:
        print(i)
    print(rover_path)
    return rover_path


ROVERPATH = main()

print("---\nChecking Distance")

drone = LatLon(40.61865173982036, -74.56913024979528, datum=Datums.NAD83)
for i in range(0,len(ROVERPATH)):
    waypoint = LatLon(ROVERPATH[i][0],ROVERPATH[i][1], datum=Datums.NAD83)
    print(waypoint)
    print(drone.distanceTo(waypoint))



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