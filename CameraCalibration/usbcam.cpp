#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
using namespace std::chrono;
using namespace cv;
using namespace std;

//https://answers.opencv.org/question/186236/accessing-a-usb-connected-camera/


camera = cv2.VideoCapture(1) // Where i ranged from 0 to 99)
camera = cv2.VideoCapture("TCA-9 USB2.0 Camera")
camera = cv2.VideoCapture("Port_#0001.Hub_0001")
