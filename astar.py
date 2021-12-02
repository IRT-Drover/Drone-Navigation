'''python astar work in progress'''
import cv2 as cv
import numpy as np
import math
import sys
import heapq # priority heap queue

global INFINITY
INFINITY = 9999999

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, x=None, y=None):
        self.parent = parent
        self.x = x
        self.y = y

        self.g = INFINITY
        self.h = INFINITY
        self.f = INFINITY

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str(self.x) + " " + str(self.y)


def isValid(node, image):
    if((node.x >= 0) and (node.x < image.shape[0]) and (node.y >= 0) and (node.y >= image.shape[1]):
        return False 
    return True 
    

def calculateG(parentNode, childNode, image):
'''set the cost of the current node as the color difference of its parent using the image ndarray'''
    parentBGR = image[parent.x][parent.y]
    childBGR = image[child.x][child.y]
    childNode.g = parentNode.g + math.sqrt((parentBGR[0]-childBGR[0])**2 + (parentBGR[1]-childBGR[1])**2 + (parentBGR[2]-childBGR[2])**2) # difference between color values as 3d points

def calculateH(node, endNode):
'''calculates the distance from the current node to the final node'''
    node.h = (((node.x - endNode.x) ** 2) + ((node.y - endNode.y) ** 2)) ** 0.5

def isDestination(node, endNode):
'''determines whether the current node is the destination'''
    return node.x == endNode.x and node.y == endNode.y

def aStar(startNode, endNode, image):
    cols = image.shape[0]
    rows = image.shape[1]
    openQueue = [] # HeapQueue for discovered nodes that may need to be visited
    





    

    






