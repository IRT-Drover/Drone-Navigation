# astar algorithm pathfinding
# input is an image called maze.png
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

    def __hash__(self):
        string = str(self.x) + "," + str(self.y)
        return hash(string)


def isValid(node, image):
    '''checks if a node's coordinates exist on an image ndarray'''
    if node.x < 0 or node.y < 0:
        return False
    if node.x >= image.shape[0] or node.y >= image.shape[1]:
        return False
    return True


def calculateG(parentNode, childNode, image, isDiagonal):
    '''return the cost of the child node as the color difference of its parent using the image ndarray'''
    parentBGR = image[parentNode.x][parentNode.y]
    # parentBGR = parentBGR.astype('int8')
    childBGR = image[childNode.x][childNode.y]
    # childBGR = childBGR.astype('int8')
    if not isDiagonal:
        g = 1 + parentNode.g + math.sqrt((parentBGR[0]-childBGR[0])**2 + (parentBGR[1]-childBGR[1])**2 + (parentBGR[2]-childBGR[2])**2) # difference between color values as 3d points
    else:
        g = math.sqrt(2) + parentNode.g + math.sqrt((parentBGR[0]-childBGR[0])**2 + (parentBGR[1]-childBGR[1])**2 + (parentBGR[2]-childBGR[2])**2) # difference between color values as 3d points
    # g = parentNode.g + 1
    return g

def calculateH(node, endNode):
    '''calculates the distance from the current node to the final node'''
    h = (((node.x - endNode.x) ** 2) + ((node.y - endNode.y) ** 2)) ** 0.5
    return h

def reconstructPath(node):
    '''returns list of subsequent parent nodes of specified node'''
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    return path[::-1] ### Return reversed path, is there a faster way to do this

def aStar(startNode, endNode, image):
    cols = image.shape[0]
    rows = image.shape[1]

    openQueue = [] # Priority queue for discovered nodes that may need to be visited (priority is F value of node)
    closedSet = set() # Hash set for visited nodes that should not be revisited (equality is based on coordinates)

    startNode.g = 0
    startNode.h = calculateH(startNode, endNode)
    startNode.f = startNode.h
    heapq.heappush(openQueue, startNode)

    while (len(openQueue) != 0):
        currentNode = heapq.heappop(openQueue) # node with the smallest f value

        if currentNode in closedSet:
            continue

        closedSet.add(currentNode)
        # print(currentNode)

        if currentNode == endNode: # reached the end
            return reconstructPath(currentNode)

        #for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # adjacent nodes, not including diagonals
        for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # adjacent nodes including diagonals

            nodeX = currentNode.x + newPosition[0]
            nodeY = currentNode.y + newPosition[1]
            child = Node()
            child.x = nodeX
            child.y = nodeY
            # print("Child: ", child)
            
            if not isValid(child, image):
                continue

            # if np.array_equal(image[child.x, child.y], (0,0,0)): # black pixel
            #     continue

            if child in closedSet: # node has been visited
                continue


            if newPosition[0] == 0 or newPosition[1] == 0: # adjacent
                g = calculateG(currentNode, child, image, False)
            else: # diagonal
                g = calculateG(currentNode, child, image, True)
            h = calculateH(child, endNode)

            child.parent = currentNode
            child.g = g
            child.h = h
            child.f = g + h
            
            heapq.heappush(openQueue, child)
        
                
    return "failure"
    
                    
def main():
    img = cv.imread(cv.samples.findFile("maze.png")) # img is a numpy ndarray
    img = img.astype('int8') # cast to signed integer so subtraction will not result in overflow, THIS USES SIGNIFICANTLY MORE MEMORY AND RUNTIME BUT IS NECCESARY FOR COLORS
    img2 = cv.imread(cv.samples.findFile("maze.png")) # img is a numpy ndarray
    
    if img is None:
        sys.exit("Could not read the image.")
    
    #img = img.tolist() # row list of column list of pixel RGB list
    #cv.imshow("img.png", img2)

    start = Node()
    start.x = 0
    start.y = 0
    end = Node()
    end.x = len(img)-1
    end.y = len(img[len(img)-1])-1

    # print("Start: ", start)
    # print("End: ", end)

    path = aStar(start, end, img)
    for coords in path:
        # print("COORDS:" + str(coords))
        img2[coords.x, coords.y] = [0,0,255]
    
    cv.imwrite("test.png", img2)


main()
