
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

    # in aStar, parent should be the node immediately preceeding the node on the cheapest known path

    def __init__(self, parent=None, x=None, y=None):
        self.parent = parent
        self.x = x
        self.y = y

        self.g = INFINITY
        self.h = INFINITY
        self.f = INFINITY

    def __eq__(self, other):
        # overload == operator for coordinates
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str(self.x) + " " + str(self.y)

    def __lt__(self, other):
        # overload < operator for f value; this is needed for the heapQueue in aStar algorithm
        return self.f < other.f


def isValid(node, image):
    '''checks if a node's coordinates exist on an image ndarray'''
    if((node.x >= 0) and (node.x < image.shape[0]) and (node.y >= 0) and (node.y >= image.shape[1])):
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

def reconstructPath(node):
    '''returns list of all parent nodes of specified node'''
    path = []
    while node is not None:
        path.append(current)
        current = current.parent
    return path[::-1] # Return reversed path

def aStar(startNode, endNode, image):
    cols = image.shape[0]
    rows = image.shape[1]

    openQueue = [] # HeapQueue for discovered nodes that may need to be visited

    startNode.g = 0
    startNode.h = calculateH(startNode, endNode)
    startNode.f = startNode.h
    openQueue = heapq()
    openQueue.append(startNode)

    while (len(openQueue) != 0):
        currentNode = openQueue.heappop() # node with the smallest f value

        if currentNode == endNode: # reached the end
            return reconstructPath(current)
        
        children = []
        for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # adjacent nodes
            nodeX = current.x + newPosition[0]
            nodeY = current.y + newPosition[1]
            child = Node()
            child.x = nodeX
            child.y = nodeY
            print("Child: ", child);
            
            if not isValid(child, image):
                continue    

            child.g = current.g + calculateG(current, child, image)
            child.h = calculateH(child, endNode)
            child.f = child.g + child.h
            
            if child not in openQueue:
                heapq.heappush(openQueue, child)
                
    return "failure"
                    
                
            
            

   
        

        

def main():
    img = cv.imread(cv.samples.findFile("SimpleObstacle.png")) # img is a numpy ndarray
    img2 = cv.imread(cv.samples.findFile("SimpleObstacle.png")) # img is a numpy ndarray

    
    if img is None:
        sys.exit("Could not read the image.")
    
    #img = img.tolist() # row list of column list of pixel RGB list
    cv.imshow("img.png", img2)

    start = Node()
    start.x = 0
    start.y = 0
    end = Node()

    end.x = len(img)-1
    end.y = len(img[len(img)-1])-1

    print("Start: ", start)
    print("End: ", end)

    path = aStar(start, end, img)
    print(path)
    # print(path)
    for coords in path:
        img2[coords.y, coords.x] = [0,0,255]
    
    cv.imwrite("test.png", img2)
    



main()
