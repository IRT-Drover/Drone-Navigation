//Author - Diego
//Version 1.0.1

//Imports for neccesary libraries. 
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <set>
#include <iterator>
#include <list> 
#include <iterator>
#include <vector> 
#include <algorithm>
#include <time.h> 


//This global variable infinite will be used for the cost values and the dijstra algorithmn
#define INFINITY 999999

using namespace cv;
using namespace std;


const char* keys = { "Error" };

struct node {
    int parentX;
    int parentY;
    double f;
    double g;
    double h;
};

// the pixel has three values(f, x, y). They will be sorted in a set based on their f values.
typedef pair<double, pair<int, int>> pixelValues;

void showImage(string imageName, Mat image)
{
    namedWindow(imageName, WINDOW_AUTOSIZE);

    imshow(imageName, image);
}


//The to_index function takes coordinates and returns the number of the element in the array from those coordinates.
// So, 0,0 would be 0 ... 0, 1 would be 1... etc
int to_index(int x, int y, Mat image)
{
    return x * image.cols + y;
}


//To coordinates does the reverse of to_index. It takes the number of element in the array and returns its two coordinates.
void to_coordinates(int pixelIndex, int coordinate[], Mat image)
{
    coordinate[0] = int(pixelIndex / image.cols);       // division full part only   example: 10//3 -> 3 ;   20//3 = 6
    coordinate[1] = int(pixelIndex % image.cols);      // division remainder only
}

//to check if the node is valid
bool isValid(int row, int col, Mat image)
{
    if (((row >= 0) && (row < image.rows) && (col >= 0) && (col < image.cols)))
    {
       return true;
    }

    else
    {
       return false;
    }
}

float calculateG(int x, int y, int xDiff, int yDiff, int previousGVal, Mat image)
{
    Vec3b pixel1 = image.at<Vec3b>(Point(x, y));
    Vec3b pixel2 = image.at<Vec3b>(Point(x + xDiff, y + yDiff));
    int cost = 0;

    for (int i = 0; i < 3; i++)
    {
        cost += abs(pixel1[i] - pixel2[i]);
    }

    cost += previousGVal;
    return cost;
}

bool isDestination(int cell[], int dest[])
{
    if (cell[0] == dest[0] && cell[1] == dest[1])
    {
        return true;
    }

    return false;
}

double calculateH(int cell[], int end[])
{
    //printf("firstVal = %d \n", (end[0] - cell[0]) * (end[0] - cell[0]));
    //printf("secondVal = %d \n", (end[1] - cell[1]) * (end[1] - cell[1]));
    //printf("cellX = %d , cellY = %d \n", cell[0], cell[1]);
    //printf("endX = %d , endY = %d \n", end[0], end[1]);
    return sqrt(((end[0] - cell[0]) * (end[0] - cell[0])) + ((end[1] - cell[1]) * (end[1] - cell[1])));
}


vector<vector<node>> aStar(int start[], int end[], Mat image)
{
    const int rows = image.rows;
    const int cols = image.cols;
    vector<vector<bool>> closed( rows, vector<bool>(cols, false));
    vector<vector<node>> nodeArray( rows, vector<node>(cols));

    int i = start[0];
    int j = start[1];

    // set the f, g, and h values in the node array to infinity
    // set the parent x and y values to -1
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            nodeArray[i][j].parentX = -1;
            nodeArray[i][j].parentY = -1;
            nodeArray[i][j].f = INFINITY;
            nodeArray[i][j].g = INFINITY;
            nodeArray[i][j].h = INFINITY;
        }
    }

    nodeArray[start[0]][start[1]].f = 0;
    nodeArray[start[0]][start[1]].g = 0;
    nodeArray[start[0]][start[1]].h = calculateH(start, end);
    
    set<pixelValues> openList;

    openList.insert(make_pair(0.0, make_pair(i, j)));

    bool keepGoing = true;

    int directions[8][2] = { {0, 1}, {0, -1}, {1, 0}, {1, 1}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1} };

    while (!openList.empty())
    {
        pixelValues pixel = *openList.begin();
        openList.erase(openList.begin());

        i = pixel.second.first;
        j = pixel.second.second;
        
        closed[i][j] = true;

        float newG;
        float newH;
        float newF;
        
        //And the third is to compare the pixel at (i, j)to all 8 other ones around it
        for (int h = 0; h <= 8; h++)  // 8 is the total number of feasible directions
        {
            //First the x and y diffs are found
            int xDiff = directions[h][0];
            int yDiff = directions[h][1];
            int cell[] = { i + xDiff, j + yDiff };

            if(isValid(cell[0], cell[1], image))
            {
                if(isDestination(cell, end))
                {
                    printf("Reached Destination");
                    nodeArray[cell[0]][cell[1]].parentX = i;
                    nodeArray[cell[0]][cell[1]].parentY = j;
                    keepGoing = false;
                    return nodeArray;
                }

                else if (closed[cell[0]][cell[1]] == false)
                {
                    newG = calculateG( i, j, xDiff, yDiff, nodeArray[i][j].g, image);
                    newH = calculateH(cell, end);
                    //printf("costH = %f \n", newH);
                    newF = newG + newH;


                    if (nodeArray[cell[0]][cell[1]].f == INFINITY || nodeArray[cell[0]][cell[1]].f > newF)
                    {
                        openList.insert(make_pair(newF, make_pair(cell[0], cell[1])));
                        nodeArray[cell[0]][cell[1]].f = newF;
                        /*printf("costF = %d \n", newF);
                        printf("costG = %d \n", newG);
                        printf("costH = %d \n", newH);*/
                        nodeArray[cell[0]][cell[1]].g = newG;
                        nodeArray[cell[0]][cell[1]].h = newH;
                        nodeArray[cell[0]][cell[1]].parentX = i;
                        nodeArray[cell[0]][cell[1]].parentY = j;
                        //printf("x1 = %d , y1 = %d \n", cell[0], cell[1]);
                        //printf("x2 = %d, y2 = %d \n", i, j);
                    }
                }
            }

        }
    }

    if (keepGoing == true)
    {
        printf("Destination not found");
    }

    return nodeArray;
}

Mat tracePath(int start[], int end[], vector<vector<node>> nodeArray, Mat image)
{
    bool keepGoing = true;
    vector<vector<int>> path;
    path.push_back({end[0], end[1]});
    int parentX = nodeArray[end[0]][end[1]].parentX;
    int parentY = nodeArray[end[0]][end[1]].parentY;
    int newParentX;
    int newParentY;
    //printf("x = %d , y = %d \n", parentX, parentY);
    Mat tracedImage = image;
    //printf("xStart = %d , yStart = %d \n", start[0], start[1]);
    //printf("x = %d , y = %d \n", nodeArray[11][10].parentX, nodeArray[11][10].parentY);
    int counter = 0;
    while (keepGoing)
    {
        counter++;
        path.push_back({ parentX, parentY });
        newParentX = nodeArray[parentX][parentY].parentX;
        newParentY = nodeArray[parentX][parentY].parentY;
        //printf("x = %d , y = %d \n", newParentX, newParentY);
        Vec3b& color = tracedImage.at<Vec3b>(newParentY, newParentX);
        color[0] = 255;
        color[1] = 0;
        color[2] = 255;
        //tracedImage.at<Vec3b>(Point(newParentY, newParentX)) = color;

        if (newParentX == start[0] && newParentY == start[1])
        {
            keepGoing = false;
        }

        parentX = newParentX;
        parentY = newParentY;
    }
    return tracedImage;
    //for (int i = 0; i < path.size(); i++)
    //{
    //    printf("x = %d , y = %d \n", path[i][0], path[i][1]);
    //}
}


int main(int arfc, const char** argv)
{	
	Mat image;
	const char* filename1 = "C:\\Users\\diego\\source\\repos\\road.png";
	image = imread((filename1), IMREAD_COLOR);

    if (image.empty())
    {
        //printf("Cannot read image file: %s\n", filename.c_str());
        printf("Cannot read image file: %s\n", filename1);
        return -1;
    }
    int start[] = { 10, 190 };
    int end[] = { 200, 50 };

    //printf("gVal = %f \n", calculateG( 10,  10, 20, 20, 0, image));
    /*printf("gVal = %f \n", calculateH(start, end));
    printf("starting");*/
    vector<vector<node>> nodearray = aStar(start, end, image);

    printf("start tracing");
    Mat tracedimage = tracePath(start, end, nodearray, image);

    showImage("tracedimage", tracedimage);
    //showimage("testimage", image);
    waitKey(0);
    destroyAllWindows();
}
