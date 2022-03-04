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

def distance(pixHome, pix1, resolution, magnification):
    distance_img_y = (pixHome[1] - pix1[1]) / resolution
    distance_map_y = distance_img_y / magnification
    print("Vert meters to center: " + str(distance_map_y))
    
    distance_img_x = (pix1[0] - pixHome[0]) / resolution
    distance_map_x = distance_img_x / magnification
    print("Horiz meters to center: " + str(distance_map_x))
    
    return [distance_map_x, distance_map_y]

def main(PATH, directory):
    #Camera Specs
    # # focal length calculated from info given by manufacture
    # focal = 0.002 # focal length in meters #x=sensor_hdimens*pixelsize_in_micrometers/1,000,000 #f = x/(2tan(angleofview/2))
    # unitcell = 1.12 # size of single pixel in micrometers
    # resolution = 1/unitcell*1000000 # pixels per meter
    # sensor_H = 4656 # pixel dimensions of camera sensor/photo
    # sensor_V = 3496
    
    npz_calib_file = np.load('CameraCalibration/calibration_data.npz')
    intrinsic_matrix = npz_calib_file['intrinsic_matrix']
    unitcell = 3 # size of single pixel in micrometers
    #focal = 0.0035 # focal length by taking an image and calculated necessary focal length to produce an accurate real world distance
    # AVG focal length from camera calibration
    focal = (intrinsic_matrix[0][0]*unitcell + intrinsic_matrix[1][1]*unitcell)/2/1000000 - 0.000117# focal length in meters #x=sensor_hdimens*pixelsize_in_micrometers/1,000,000 #f = x/(2tan(angleofview/2))
    resolution = 1/unitcell*1000000 # pixels per meter
    print("focal length: " + str(focal))
    
    img = cv.imread(f'{directory}Distance_Testing-constantobjsize1.png')
    sensor_H = img.shape[1] # pixel dimensions of camera sensor/photo (should be the same thing)
    sensor_V = img.shape[0]
    
    print(sensor_H)
    print(sensor_V)
    print("sensor dimensions in mm:")
    print(sensor_H/resolution*1000) # in mm
    print(sensor_V/resolution*1000) # in mm
    
    #TESTING PATH. ALL CORNERS AND MIDPOINTS, STARTING TOP LEFT
    # PATH = []
    # for i in [[0,0],[sensor_H/2-1,0],[sensor_H-1,0],[sensor_H-1,sensor_V/2-1],[sensor_H-1,sensor_V-1],[sensor_H/2-1,sensor_V-1],[0,sensor_V-1], [0,sensor_V/2-1]]:
    #     testpixel = Node()
    #     testpixel.x = i[0]
    #     testpixel.y = i[1]
    #     PATH.append(testpixel)
    
    altitude = 1.16 # in meters
    image_dist = (altitude * focal) / (altitude - focal)
    magnification = image_dist/altitude
    print("image distance: " + str(image_dist))
    print("altitude: " + str(altitude))
    print("magnification: " + str(magnification))
    
    # image_center = Node()
    # pixel_y_0 = int((len(img))/2-1) # pixel y-coordinate
    # pixel_x_0 = int((len(img[0]))/2-1) # pixel x-coordinate 
    pixel_y_0 = int((sensor_V)//2) # pixel y-coordinate
    pixel_x_0 = int((sensor_H)//2) # pixel x-coordinate 
    
    lat_0 = 40.61865173982036 #latitude
    long_0 = -74.56913024979528 #longitude
    drone = LatLon(40.61865173982036, -74.56913024979528, datum=Datums.NAD83) # default datum is WGS-84

    rover_path = []
    
    print("---")
    
    # TWO SETS OF PIXEL COORDINATES FOR CALCULATING DISTANCE
    # points1 = [(1200,845),(1022,659), (924,559), (870,502), (828,460), (807,436), (790,417)] # COMMENT OUT
    points1 = [(1204,861), (1091,795), (1012,747), (965,718), (915,682), (869,656), (838,637), (821,624), (801,613)]
    pointsHome = [] # COMMENT OUT
    for i in range(0,len(points1)):
        pointsHome.append((pixel_x_0,points1[i][1]))
    print(pointsHome)
    # ALTITUDE FOR THE DIFFERENT DISTANCES
    # altitude = [1.774, 2.685, 3.595, 4.495, 5.421, 6.336, 7.25] # COMMENT OUT
    altitude = [1.98, 2.47, 2.98, 3.45, 4.145, 4.865, 5.555, 6.265, 6.945] # COMMENT OUT
    # Finds the distance between any two points on an image and draws a line and adds distance to image
    avg_dist = 0
    for i in range(0,len(points1)):
        image_dist = (altitude[i] * focal) / (altitude[i] - focal)
        magnification = image_dist/altitude[i]
        
        print('Calculating distance between... ' + str(pointsHome[i]) + " and " + str(points1[i]))
        distance_map_x, distance_map_y = distance(pointsHome[i], points1[i], resolution, magnification)
        distance_map = math.sqrt(distance_map_x**2 + distance_map_y**2)
        cv.line(img, pointsHome[i], points1[i], (0, 255, 0), 3)
        
        # displaying distance on image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(distance_map) + " meters-->",
                   (pointsHome[i][0]-300, pointsHome[i][1]+5),
                   font, 0.5, (255, 0, 0), 2)
        avg_dist += distance_map
    avg_dist = avg_dist/len(points1)
    print('\nAverage calculated distance: ' + str(avg_dist))
    print('---')
    cv.imshow('Distance between Points',img)
    cv.waitKey(0)
    cv.imwrite(f'{directory}Distance_Testing-constantobjsize1-dist_measurements.png', img)
    cv.destroyAllWindows()

    # Finds physical x- y-distance to a point and returns the GPS coordinates
    for i in range(0,len(PATH)):
        pixelCenter = [pixel_x_0, pixel_y_0]
        # image_dist = (altitude[i] * focal) / (altitude[i] - focal) # DELETE
        # magnification = image_dist/altitude[i] # DELETE
        
        distance_map_x, distance_map_y = distance(pixelCenter, [PATH[i].x, PATH[i].y], resolution, magnification)
        # distance_map_x, distance_map_y = distance(pixelCenter, [pixel.x, pixel.y], resolution, magnification)
        
        # distance_img_y = (pixel_y_0 - pixel.y) / resolution
        # distance_map_y = distance_img_y / magnification
        # print("Vert meters to center: " + str(distance_map_y))
        
        # distance_img_x = (pixel.x - pixel_x_0) / resolution
        # distance_map_x = distance_img_x / magnification
        # print("Horiz meters to center: " + str(distance_map_x))
        
        distance_map = math.sqrt(distance_map_x**2 + distance_map_y**2)
        
        bearing = 0 # compass 360 degrees; north = 0 degrees
        if distance_map_x != 0:
            bearing = 90 - (math.atan(distance_map_y/distance_map_x)*180/math.pi)
            if distance_map_x < 0:
                bearing += 180
        elif distance_map_y < 0:
            bearing = 180
        
        print("\nBearing: " + str(bearing))
        print("Distance: " + str(distance_map) + "\n")
        # http://www.movable-type.co.uk/scripts/latlong-vincenty.html
        # Millimeter accuracy. To get to nanometer accuracy, will have to switch from Vincenty to Karney's method
        waypoint = drone.destination(distance_map, bearing)
        
        rover_path.append([waypoint.lat, waypoint.lon])
    
    print("Drone Coordinate: ")
    print(drone)
    print([lat_0,long_0])
    print("Path:")
    for wp in rover_path:
        print(wp)
        
    return rover_path

directory = 'CameraCalibration/pixeltocoordinate_imagetesting/'
PATH = []
img = cv.imread(f'{directory}Distance_Testing-constantobjsize1.png')
sensor_H = img.shape[1] # pixel dimensions of camera sensor/photo (should be the same thing)
sensor_V = img.shape[0]
for i in [[0,0],[825,460],[sensor_H, sensor_V//2]]:
    testpixel = Node()
    testpixel.x = i[0]
    testpixel.y = i[1]
    PATH.append(testpixel)
ROVERPATH = main(PATH, directory)

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