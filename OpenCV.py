
import cv2
from datetime import datetime
import os
global vid, path, cam, today, now, time, newpath
path = "/home/pi/Drone-Navigation/DronePictures"
cam = cv2.VideoCapture(0)
class OpenCV:
    
    #today = datetime.today()
    now = datetime.now()
    day = str(now.date())
    time = str(now.time())     
    newPath = r"/home/pi/Drone-Navigation/DronePictures/" + day + " " + time
    os.makedirs(newPath)
    path = newPath
    counter = 0
    repository = open(newPath + "/pictureData.txt", "a")
    
    def takePicture(self, location, alt):
        ret, img = cam.read()
        self.counter += 1
        cv2.imwrite(self.newPath + "/Drone Picture: " + self.day + " " + self.time + ".png", img)
        self.repository.write("Picture " + str(self.counter) + "\n" + "Location " + str(location) + "\n" + "Altitude" + str(alt))

        
    