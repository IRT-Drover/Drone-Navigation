import glob
import numpy as np
import math
from pygeodesy.ellipsoidalVincenty import LatLon
from pygeodesy import Datums

# print(glob.glob('*[!s].sh'))
# #print(glob.glob('*.pyc', exclude = ['HaltingProblemTest.java']))

# print(glob.glob('/Users/charlesjiang/Documents/GitHub/*'))

# for i in range(1,2):
#     print('a')
    
# gpsdata = np.load('DronePictures/2021-04-19 18:08:11.670318/GPSDATA_TOSEND.npz')

# for i in gpsdata['GPSPATHS']:
#     for j in i:
#         print(j)


# GPSPATHS = {}
# for img_num in range(1, 4):
#     GPSPATHS[img_num] = [1+img_num,2+img_num,3+img_num]

# np.save('data_by_image_package', GPSPATHS)

received = np.load('numpytest.npy', allow_pickle='TRUE').item()
print(received)
print(received[1])

received["Picture 4"] = [1,2,3,4,5,6,7,89,9]
print(received)

np.save('numpytest.npy', received)
received2 = np.load('numpytest.npy', allow_pickle='TRUE').item()
print(received2)

# GPS_PATH = [1+img_num,2+img_num,3+img_num]
# GPSPATHS = np.array([(img_num, GPS_PATH)],
#                             dtype=[('name','O'), ('path','O')])
# print(GPSPATHS)
# print(GPSPATHS[0])
# print(GPSPATHS[0][1])

# GPSPATHS = np.append(GPSPATHS, np.array([(1, [1,2,3])],
#                             dtype=GPSPATHS.dtype))

# print(GPSPATHS[1])

GPSPaths = np.load('DronePictures/2022-06-05 Satellite Image Testing/GPSDATAPACKAGE.npy', allow_pickle='TRUE').item()
print(GPSPaths.keys())
# GPSPaths.pop(('Picture', 1))
# np.save('DronePictures/2022-06-05 Satellite Image Testing/GPSDATAPACKAGE.npy', GPSPaths)

# drone = LatLon(0, 0, datum=Datums.NAD83) # default datum is WGS-84
# bearing = 0 # compass 360 degrees; north = 0 degrees
# distances = [[10,0], [0,10], [-10,0], [0,-10], [10,10], [-10,10], [-10,-10], [10,-10], [-1,10]]

# for i in range(0,len(distances)):
#     if distances[i][0] != 0:
#         bearing = 90 - (math.atan(distances[i][1]/distances[i][0])*180/math.pi) # accounts for angle of top of image from North
#         if distances[i][0] < 0:
#             bearing += 180
#     elif distances[i][1] < 0:
#         bearing = 180
#     print('bearing v1: ' + str(bearing))
#     print(drone.destination(math.sqrt(distances[i][0]**2 + distances[i][1]**2), bearing))
    
# print('----')    

# bearing = 0 # compass 360 degrees; north = 0 degrees # accounts for angle of image from north
# for i in range(0,len(distances)):
#     if distances[i][0] != 0:
#         bearing = 90 - (math.atan(distances[i][1]/distances[i][0])*180/math.pi) + 30
#         if distances[i][0] < 0:
#             bearing += 180
#     elif distances[i][1] < 0:
#         bearing = 180 + 30