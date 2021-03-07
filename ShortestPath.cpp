
//Author - Diego
//Version 1.0.0
//1-3-2021

//This program takes an image a finds the canny edges of that image.
//It then takes a start node and an end node in order to find the shortest path to those around the canny edges with dijstra's algorithmn
// I used the open cv library for c++. Download instructions are below: 
// how to install the opencv libraries https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html 
// Pre-built libs: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/ 
// set environmental variables https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path 
//how to import the opencv libs in visual studio: https://docs.opencv.org/master/dd/d6e/tutorial_windows_visual_studio_opencv.html 

//Imports for neccesary libraries. 
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <list> 
#include <iterator>
#include <vector> 
#include <algorithm>
#include <time.h> 

//This global variable infinite will be used for the cost values and the dijstra algorithmn
#define INFINITE 999999

using namespace cv;
using namespace std;


const char* keys = { "Error" };

//***********************************************************************************************************
//***********************************************************************************************************


//Shows the image window
static void print_images(string window_name, Mat img)
{

    // Create a window
    namedWindow(window_name, WINDOW_AUTOSIZE);

    // create a trackbar for modifying images
    //createTrackbar("Original image", window_name, edgeThresh, 1, processImageIntoCannyEdges);
    //createTrackbar("Canny threshold default", window_name1, edgeThresh, 100, processImageIntoCannyEdges);
    //createTrackbar("Canny threshold Scharr", window_name2, edgeThreshScharr, 400, processImageIntoCannyEdges);

    //displaying original image
    imshow(window_name, img);

}


//Makes the median color value to change the minimum and maximum color changes needed in order to show a canny edge.
double median2(Mat img1)
{
    // Function for caclulating the median of an image matrix
    // receives an image, converts it into a vector
    // sorts the vector, and identified the median.

    int total_values = img1.rows * img1.cols;

    double med = 0;

    vector<int> vec(total_values);

    //The function goes through the whole matrix of the image and adds each value to a vector
   
    int counter = 1;
    for (int i = 1; i <= img1.rows - 1; i++)
    {
        for (int j = 1; j <= img1.cols - 1; j++)
        {
            vec[counter] = float(img1.at<uchar>(i, j));
            counter++;
        }
    }
   
    //The vector is then sorted, and if the amount of values is odd then the median is the middle number, 
    //If the amount of values is even, then the average of the middle two numbers is the median.
    sort(vec.begin(), vec.end());

    if ((total_values) % 2 != 0)
    {
        med = double(vec[total_values / 2]);
    }
    else
    {
        med = double((vec[(total_values - 1) / 2] + vec[total_values / 2]) / 2.0);
    }
    return med;

}
//***********************************************************************************************************
//ProcessImageIntoCannyEdges takes a colored image and takes the canny edges and the gray version of the picture. 
//It then shows the original picture, the black and white picture, and the picture with canny edges. 
void processImageIntoCannyEdges(bool show, Mat image, int start[], int end[], Mat& grayin, Mat& cedgein)
{
    printf("**************Starting to run:  processImageIntoCannyEdges *************   \n");
    // define a processImageIntoCannyEdges callback
    // Variables
    Mat blurImage, edge1;
    int edgeThresh = 1;
    int edgeThreshScharr = 1;

    //CREATE AN IMAGE by size and type
    //cedge.create(image.size(), image.type());
    //gray.create(image.size(), image.type());

    //Make the gray image:
    cvtColor(image, grayin, COLOR_BGR2GRAY);

    // Calculating Median for the gray image
    double v;
    int gray_cols_number = grayin.cols;
    int gray_rows_number = grayin.rows;
    int gray_total = gray_cols_number * gray_rows_number;

    //The min and max values are calculated to turn the picture into the one with canny edges.
    v = median2(grayin);

    int calc_lower = int(0.66 * v);
    int calc_upper = int(1.33 * v);
    int lower = max(0, calc_lower);
    int upper = min(255, calc_upper);
    printf("Values for canny calculation (median, lower, upper):  %f5, %i, %i \n", v, lower, upper);

    //I'm not sure why the python script dilated the image, but the code is below
    //blur(grayin, blurImage, Size(3, 3));
    // Run the edge detector on grayscale
    //Canny(blurImage, edge1, edgeThresh, edgeThresh * 3, 3);
    // documentation : https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html 

    //Turns the picture into canny edges
    Canny(grayin, cedgein, lower, upper);

    //edge1 = Scalar::all(0);
    //image.copyTo(edge1, cedgein);

     
    
    printf("**************End of run:  processImageIntoCannyEdges *************   \n");

}

//prints the Adjacency Array
void print_adjacencyArray(const vector<vector<float>>& adjacencyImageArray)
{
    //vector is sent into this function as a constant reference and it cannot be changed
    printf("**************Starting to run:  print array *************   \n");
    int printed = 0;
    int counter_below_threshold = 0;
    int counter_above_threshold = 0;

    //Cycles through the array and prints the adjacency Matrix Value
    int counter_threshold = INFINITE;
    
    for (int i = 0; i < (adjacencyImageArray.size()); i++)
    {
        for (int j = 0; j < adjacencyImageArray[i].size(); j++)
        {
            // printing only values greater than 10

            if ((abs(adjacencyImageArray[i][j]) < counter_threshold) && (int(abs(adjacencyImageArray[i][j])) > 0))
            {
                printf(" (i,j) (%i,%i) and Cost: %f2 \n", i, j, adjacencyImageArray[i][j]);
                printed = 1;
                counter_above_threshold++;
            }
        }
        if (printed == 1)
        {
            printf("\n");
            printed = 0;
            counter_below_threshold++;
        }
    }
    printf("counter_threshold = %i, counter_above_threshold = %i, counter_below_threshold = %i", counter_threshold, counter_above_threshold, counter_below_threshold);
    printf("**************end of run:  print array *************   \n");
}

//The to_index function takes coordinates and returns the number of the element in the array from those coordinates.
// So, 0,0 would be 0 ... 0, 1 would be 1... etc
int to_index(int x[], Mat img)
{
    return x[0] * img.cols + x[1];
}


//To coordinates does the reverse of to_index. It takes the number of element in the array and returns its two coordinates.
void to_coordinates(int pixel_index_calc, int coordinate[], Mat img)
{
    coordinate[0] = int(pixel_index_calc / img.cols);       // division full part only   example: 10//3 -> 3 ;   20//3 = 6
    coordinate[1] = int(pixel_index_calc % img.cols);      // division remainder only
}


//Bitmap_to_adjacency makes a cost matrix (adjacency matrix) using the gray image matrix made in processCannyImage.
void bitmap_to_adjacency(Mat gray, vector<vector<float>>& adjacencyImageArray)
{
    //printf("**************Starting to run:  bitmap_to_adjacency *************   \n");
    // A sparse adjacency matrix.
    // Two pixels are adjacent in the graph if both are not painted.
    // sparse matrix : https://en.wikipedia.org/wiki/Sparse_matrix
    // cpp code: http://arma.sourceforge.net/armadillo_lncs_2018.pdf 

    // The following lines fills the adjacency matrix by the directions of possible paths or allowed moves in x,y axis.
    // The x and y changes are respective to the pixel being compared to
    int directions[8][2] = { {0, 1}, {0, -1}, {1, 0}, {1, 1}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1} };

    //New variables are now made for the nested for loop.
    Mat img;
    gray.copyTo(img);

    //The x and y diff will come from the directions list to represent which pixel is being compared to the middle pixel.
    // e.g. if the x diff is x diff is 0 and the y diff is 1. The pixel will be compared to its neighbor above.
    int y_diff = 0;
    int x_diff = 0;

    //The content variables will store the value of the color from the gray matrix of the two pixel being compared.
    // Taking the absolute value of the difference between the two will give the cost of moving between the two pixels 
    float cost = 0.0;
    float content_p0 = 0.0;
    float content_p1 = 0.0;

    //These variables will store the element number of the pixels being compared.
    int adj_p0 = 0;
    int adj_p1 = 0;

    //These lists will store the coordinates of the two pixels.
    int x[2], x1[2] = { 0,0 };


    //MAIN FOR LOOP
    //The adjacency matrix is made with a x and a y value of rows * columns
    //This is because rows * columns is the number all of the elements. 
    //If you want to see which pixels (nodes for dijstra) are attatched to which, you go to the row of the pixel you want to see intersect, and if the column number of that pixel is greater than 0, they intersect. 
    //The value stored there(if it is greater than 0) is the cost of moving between the two pixels.
    //printf("Calculating the cost... \n");
    //The first for loop is for rows
    for (int i = 1; i <= img.rows - 1; i++)
    {
        //The second is for columns
        for (int j = 1; j <= img.cols - 1; j++)
        {
            //And the third is to compare the pixel at (i, j)to all 8 other ones around it
            for (int h = 0; h <= 8 - 1; h++)  // 8 is the total number of feasible directions
            {
                //First the x and y diffs are found
                y_diff = directions[h][0];
                x_diff = directions[h][1];

                // type of char in image for at() depends on the type of image or matrix https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html 
                // getting the value of a location: https://answers.opencv.org/question/12963/how-to-get-pixels-value-from-a-picture/
                // The coordinates of the compared pixels are found and so are there element numbers (for the x and the y of the matrix)
                x[0] = i;
                x[1] = j;
                x1[0] = i + y_diff;
                x1[1] = j + x_diff;
                adj_p0 = to_index(x, img);
                adj_p1 = to_index(x1, img);

                //if the pixel being compared is still in bounds
                if (((i + y_diff) <= img.rows - 1) && ((j + x_diff) <= img.cols - 1))
                {
                    //Calculates the cost by subtracting the two pixels and taking the absolute value of that
                    content_p0 = float(img.at<uchar>(i, j));
                    content_p1 = float(img.at<uchar>(i + y_diff, j + x_diff));
                    cost = abs(content_p0 - content_p1);

                    //All of the values of the matrix are at default, infinity. 
                    //This is so that the dijstra algorithm will see the large cost and avoid taking those paths.
                    //If the cost calculated is less than 20, then the adjacency matrix is updated at the position of the elements being compared.
                    if (cost <= 20) 
                    {
                        adjacencyImageArray[adj_p0][adj_p1] = cost;
                    } // this is needed to highlight that if there is no cost, there is no change. 
                    //printf("AFTER --- (%i,%i) Cost calculation inside  =  %f5  ,  %i , %i \n", i,j, adjacencyImageArray[adj_p0][adj_p1], adj_p0, adj_p1);
                }

            }

        }
    }

    //printf("Press any key to continue... \n");  cin.get();
    //printf("**************End of run:  bitmap_to_adjacency *************   \n");

}

//This function calculates the shortest path from the start pixel (node) and the end pixel (node) using the dijkstra algroithmn and the adj matrix made in the bitmap_Adjacency function
void dijkstra(vector<vector<float>>& adjacencyImageArray, int startnode, int endnode, vector<int>& short_path_startnode_endnode)
{
    // Site source of part of the code: https://www.tutorialspoint.com/cplusplus-program-for-dijkstra-s-shortest-path-algorithm 
    // The code was modified to account for vectors, and optimized runtime.

    // Function: 
    // Takes the Adj matrix, which contains all the cost values - calculated in the bit to adj function.
    // Takes the starting node
    // Takes the size of matrix as parameter to iterate.
    // Creates the predecesor vector, distance vector and visited tracking vector.
    // Calculates the shortest path across "ALL NODES" in the adj matrix
    // creates a vector with the shortest path - path_startnode_endnode

    // Prints the shortest path to the node of interest - path_startnode_endnode

    //The three important vectors:
        // Predecesor Vecotr 
        // Distance Vector 
        // Visited Tracking Vector


    //initializing some variables
    int count, nextnode, i, j;
    float mindistance;
    nextnode = 0;
    int node_of_interest = endnode;  // identify the node that we are interested to determine the destination for the shortest path. 
    bool found_node_of_interest = false;
    int n = adjacencyImageArray.size();

    // n = iterations, so it can be set to total_adj_matrix_cells in order to search all the combinations of path across the full matrix
    // or it can be set to "endnode" to perform calculations until endnode

    vector<float> distance(n, 0);
    vector<int> pred(n, startnode);
    vector<float> visited(n, 0);

    //printf("Created vector distance to store the distances. Vector size: %i \n", distance.size());
    //printf("Created vector pred to store the predecessors. Vector size: %i \n", pred.size());
    //printf("Created vector visited to store the visited nodes in the algo. Vector size: %i \n", visited.size());
    //printf("Starting dijkstra in node %i \n", startnode);
    //printf("Ending dijkstra in node %i \n", endnode);
    printf("Setting up distance and predecesor \n");


    //Sets all distances to infinity or the cost to the start node if both connect
    for (i = 0; i < n; i++)
    {
        distance[i] = adjacencyImageArray[startnode][i];
        //pred[i] = startnode;  // already initialized in vector creation
        //visited[i] = 0;       // already initialized in vector creation
    }

    //The distance from the startnode to the startnode is 0
    distance[startnode] = 0;

    //If visited[node] = 1; then the node has been visited
    //This is just stating that the startnode has been visited.
    visited[startnode] = 1;

    
    count = startnode;

    printf(" Iterating across nodes \n");
    // MAIN WHILE LOOP: 
    // Question - can the following loops be confined to start and end nodes only? <- this change will reduce the running time. for example, replace n with endnode (?)
    //The while loop has two conditions: the count is less than the amount of nodes or the end node has not been found
    found_node_of_interest = false;
    while ((count < n - 1) && (!found_node_of_interest)) //Keeps looping until all nodes have been looked at or the end node has been found
    {
        
         mindistance = INFINITE;

            //Checks all nodes
            for (i = 0; i < n; i++)
            {
                // checks if a distance other than infinity has been assigned to this node

                if (distance[i] < mindistance && !visited[i]) {
                    //If that's true then minDistance is the new distance to that node and the node is set to i
                    mindistance = distance[i];
                    nextnode = i;
                }
            }

            //The visited vector is updated
            visited[nextnode] = 1;

            //Checks all elements. If the new distance is less than the previous distance to that node from the nextNode:
            for (i = 0; i < n; i++)
                if (!visited[i])
                    if (mindistance + adjacencyImageArray[nextnode][i] < distance[i]) {

                        //The distance is then updated
                        distance[i] = mindistance + adjacencyImageArray[nextnode][i];
                        //The node is the path/pred vector is updated
                        pred[i] = nextnode;
                    }

            //If the node is equal to the end node
            if (node_of_interest == nextnode)
            {
                //The boolean is set to true, and the while loop is will stop
                found_node_of_interest = true;
                //printf("Found node of interest... %i \n", visited[nextnode]);
            }

        count++;
    }
    // END OF COST CALCULATION

    printf(" Printing distance and predecesor results... including path\n");
    float final_distance = 0;
    // Search all paths without cost, ie infinite
    //Outside of the while loop, the distances for each node is updated.
    for (i = 0; i < n; i++)
    {
        if ((i != startnode))
        {
            if ((distance[i] != INFINITE) && (i == node_of_interest)) // added to ensure that it is working only on node of interest.
            {

                //printf("\nDistance start node %i to node %i  = %f2 ", startnode, i, distance[i]);
                //printf("\nPath = %i", i);
                //The final distance to that node from node 0, is set
                final_distance = distance[i];
                //The list of nodes that need to be taken to reach the endnode from the startnode is updated
                short_path_startnode_endnode.push_back(i);
                j = i;
                do {    // iterating backwards from the current node to the starting node....
                    //The previous node from i is then put into the list until it is equal to the start node
                    j = pred[j];
                    //printf("<- %i", j);
                    short_path_startnode_endnode.push_back(j);
                } while (j != startnode);

            }
        }
    }

    printf("\nReversing path .... \n");
    //The list is then reversed, so that now it shows shich nodes must be taken to reach the end node.
    reverse(short_path_startnode_endnode.begin(), short_path_startnode_endnode.end());

    // Printing path to make sure it is accurate
    printf("\nPath from startnode to endnode  with a final distance of: %f2= ", final_distance);
    //Prints the nodes
    for (i = 0; i < short_path_startnode_endnode.size(); i++)
    {
        if (short_path_startnode_endnode[i] != 0)
        {
            printf("-> %i", short_path_startnode_endnode[i]);
        }
    }
    printf("\n");


}

//Run Through Graph Takes the gray and cannied image made in processImageCanny function and runs the dijstra algorithmn to take the 
// shortest path between a start and end point defined in main. It then shows the windows of the updated pictures.
static void runThroughGraph(int source[], int target[], Mat gray, Mat cedge, Mat &ImagePath, Mat &ImagePathEdge)
{
    printf("**************Starting to run:  runThroughGraph *************   \n");
    // START OF SECTION to declare variables, arrays, etc.
    
    //Image initialization area
    //tmp will show the image with the shortest path once found
    Mat tmp, tmp_cedge, image_to_process;
        
    gray.copyTo(image_to_process);
    gray.copyTo(tmp);
    // channels of colors for the picture. used for drawing the shortest path. 
    int channels = tmp.channels();
    // tmp_cedge: Canny image to show the shortest path once found
    cedge.copyTo(tmp_cedge);
    // channels of colors for the picture. 
    int channels_tmp_cedge = tmp_cedge.channels();


    //Determines the element number of the start and end nodes/pixels
    int start = to_index(source, image_to_process);
    int end = to_index(target, image_to_process);
    int coordinate[2] = { 0,0 };

    int i;
    time_t t1, t2;   
    double seconds;

    // Calculating the # of rows and columns that the image has
    // From python : img.shape[0] = rows and img.shape[1] = columns
    // https://stackoverflow.com/questions/10274162/how-to-find-2d-array-size-in-c 
    
    
    int img_cols_number = image_to_process.cols;
    int img_rows_number = image_to_process.rows;

    //Calculates the total amount of elements in the array
    int total_image_cells = img_rows_number * img_cols_number;
    printf("Image's row number is: %i\n", img_rows_number);
    printf("Image's column number is: %i\n", img_cols_number);
    printf("Image's total rows*columns number is: %i\n", total_image_cells);

    //Printing change of coordinates from 2d to 1d:
    printf("Start coordinates (%i,%i) were converted to a single coordinate: %i \n", source[0], source[1], to_index(source, image_to_process));
    printf("End coordinates (%i,%i) were converted to a single coordinate: %i \n", target[0], target[1], to_index(target, image_to_process));
    //Just to make sure that those are correct, the inverse is printed
    to_coordinates(to_index(source, image_to_process), coordinate, image_to_process);
    printf("Single start coordinate %i was converted back to coordinates (%i,%i)\n", to_index(source, image_to_process), coordinate[0], coordinate[1]);
    to_coordinates(to_index(target, image_to_process), coordinate, image_to_process);
    printf("Single end coordinate %i was converted back to coordinates (%i,%i) \n", to_index(target, image_to_process), target[0], target[1]);

    printf("Start position for short path: %i\n", start);
    printf("End position: %i\n", end);

    // instead of creating an variable array of 2 dimensions, I used a vector of a vector.
    // https://www.geeksforgeeks.org/2d-vector-in-cpp-with-user-defined-size/

    // replacement of adjacency dok_matrix creation
    printf("Creating adjacency matrix \n\n");
    //This is the vector of the adj array.
    //All values are set to infinity by default
    //Values in the adj vector (I call it a matrix in other comments, but technically, its a vector) represent the cost of moving from the x node to the y node
    vector<vector<float>> adjacencyImageArray(total_image_cells, vector<float>(total_image_cells, INFINITE));

    //This vector will be used in the dijkstra function to list the nodes that must be taken to reach the end node from the start node
    vector<int> short_path_startnode_endnode;

    printf("Call bitmap_to_adjacency and calculating cost \n");
    //20210107
    time(&t1);
    //The adj matrix (vector) is updated with the appropriate costs for moving from pixel to pixel
    bitmap_to_adjacency(image_to_process, adjacencyImageArray);
    time(&t2);
    seconds = difftime(t2,t1);
    printf("Completed bitmap_to_adjacency and calculating cost, total time in seconds = %.f \n",seconds);
    //20210107
    //print_adjacencyArray(adjacencyImageArray, total_values);
    //printf("Press any key to continue... \n");  cin.get();

    printf("Call djikstra to calculate the shortest path \n");
    //20210107
    //Dijkstra function is then called to find the shortestpath.
    //The shortest path is put into the short_path_startnode_endnode vector
    time(&t1);
    dijkstra(adjacencyImageArray, start, end, short_path_startnode_endnode);
    time(&t2);
    seconds = difftime(t2, t1);
    printf("Completed djikstra to calculate the shortest path, total time in seconds = %.f \n", seconds);
    //20210107



    //The following code is just for debugging, and it shows the path by showing the new image
    printf("Number of channels for the image image_to_process = %d \n", channels);
    //First the total nodes to get to the end node is calculated
    int total_path_steps = short_path_startnode_endnode.size();
    printf("Number of channels for the image cedge= %d \n", channels_tmp_cedge);

    //If the amount of numbers in each element of the image array is 1(this is because black and white images only have 1 while RGB images have 3),
    //then the image matrix will be updated
    if (channels == 1 && channels_tmp_cedge == 1)
        //For the amount of nodes to take in order to reach the endnode...
        for (i = 0; i < total_path_steps; i++)
        {
            //The function takes the coordinates of the node and changes the color of the element at those coordinates in the tmp Matrix
            to_coordinates(short_path_startnode_endnode[i], coordinate, tmp);
            //printf("Iteration: %i out of %i. Coordinate %i was converted to coordinates (%i,%i)\n", i, total_path_steps, short_path_startnode_endnode[i], coordinate[0], coordinate[1]);
            tmp.at<uchar>(coordinate[0], coordinate[1]) = 145;  // set the pixel

            //Does the same thign as above, but with the tmp_cedge Matri
            to_coordinates(short_path_startnode_endnode[i], coordinate, tmp_cedge);
            tmp_cedge.at<uchar>(coordinate[0], coordinate[1]) = 145;  // set the pixel

        }
    else
    {
        //If the image has more than one channel, a error message is printed
        printf("the image has 3 channels... this above drawing is only supporting 1 channel for image_to_process images \n");
    }

    tmp.copyTo(ImagePath);
    tmp_cedge.copyTo(ImagePathEdge);

    printf("**************End of run:  runThroughGraph *************   \n");
}

static void showDifferentTypesOfImageManipulation(int source[], int target[], Mat gray_original, Mat cedge)
{
    Mat quadrant, scaled,cloning,copyImage;
   
    
    //Cropping the image to reduce the adjacency matrix calculations
    
    Rect crop_region(source[0], source[1], target[0], target[1]);
    quadrant = gray_original(crop_region);
    cloning = gray_original.clone();
    copyImage = gray_original;


    float scaleFactor = 0.5f;
    resize(gray_original, scaled, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LANCZOS4);

    
    print_images("Quadrant", quadrant); 
    print_images("cloning", cloning);
    print_images("copyImage", copyImage);
    print_images("ORIGINAL", gray_original); 
    print_images("Scaled", scaled);
    
    printf("\n\noriginal image in %i, %i, %i, %i with new shape as: %i,%i -- size %i\n", source[0], source[1], target[0], target[1], gray_original.rows, gray_original.cols, gray_original.size());
    printf("\nquadrant image in %i, %i, %i, %i with new shape as: %i,%i -- size %i\n", source[0], source[1], target[0], target[1], quadrant.rows, quadrant.cols, quadrant.size());
    printf("\ncloning image in %i, %i, %i, %i with new shape as: %i,%i -- size %i\n", source[0], source[1], target[0], target[1], cloning.rows, cloning.cols, cloning.size());
    printf("\ncopyImage image in %i, %i, %i, %i with new shape as: %i,%i -- size %i\n", source[0], source[1], target[0], target[1], copyImage.rows, copyImage.cols, copyImage.size());
    printf("\nscaled image in %i, %i, %i, %i with new shape as: %i,%i -- size %i\n\n", source[0], source[1], target[0], target[1], scaled.rows, scaled.cols, scaled.size());

    waitKey(0);

}


// MAIN PROGRAM
int main(int argc, const char** argv)
{
    //Initializes some matrix variables
    Mat image, gray, cedge, grayed_scaled, image_scaled, ImagePath, ImagePathEdge, cedge_scaled, ImagePath_scaled, ImagePathEdge_scaled;
    CommandLineParser parser(argc, argv, keys);

    //string filename = parser.get<string>(0);
    //image = imread(samples::findFile(filename), IMREAD_COLOR);

    //The filename of the location of the picture is opened
   
    const char* filename1 = "C:\\Users\\diego\\source\\repos\\Maze_1.png";
    
    //The image is read by openCV in its RGB format
    image = imread((filename1), IMREAD_COLOR);
    
    //If it cannot find the image (the image matrix will be empty or an error message will pop up), an error message is printed, and the program stop
    if (image.empty())
    {
        //printf("Cannot read image file: %s\n", filename.c_str());
        printf("Cannot read image file: %s\n", filename1);
        return -1;
    }
    
    //The showimg boolean will cause the program to either show the original image, the canny edges image, the black and white image, or if false, the program will not show these images
    bool showing = true;

    //The program then takes the colored image and makes it black and white
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // variables declaration for resizing
    bool resizing = true;

    float scaleFactor = .65;
    int scaleFactorBack = int(1 /scaleFactor);

    // The number of columns and rows for the image(black and white) is calculated
    int gray_cols_number = gray.cols;
    int gray_rows_number = gray.rows;

    //The start and end pixel's coordinates are intialized 
    int start[2] = { 10,10 };
    int end[2] = { 258, 258 };

    bool validated_start_end = false;

    // These if statement check to see if the start and end coordinates given are within the image's range (basically: are they valid)
    if (start[1] >= 0 && start[1] <= gray.rows)
        if (start[2] >= 0 && start[2] <= gray.cols)
            if (end[1] >= 0 && end[1] <= gray.rows)
                if (end[2] >= 0 && end[2] <= gray.cols)
                {
                    validated_start_end = true;
                }

    // If coordinates are within the image, then execute the program
    if (validated_start_end = true)
    {
        //showDifferentTypesOfImageManipulation(start, end, gray, cedge);
        
        if (resizing)
        {
            
            printf("Scale Factor is active and set to %f of the original image - same ratio for rows and columns axis\n", scaleFactor);
            // resizing and coordinate changes https://answers.opencv.org/question/41387/resize-image-compute-resize-back-to-original/ 
            
            resize(image, image_scaled, cv::Size(image.cols * scaleFactor, image.rows * scaleFactor));
            resize(gray, grayed_scaled, cv::Size(gray.cols * scaleFactor, gray.rows * scaleFactor));
                        
            start[0] = start[0] * scaleFactor;
            start[1] = start[1] * scaleFactor;
            end[0] = end[0] * scaleFactor;
            end[1] = end[1] * scaleFactor;
            processImageIntoCannyEdges(showing, image_scaled, start, end, grayed_scaled, cedge);
            
            //Canny returns the c_edge therefore it needs to be scaled as well:
            resize(cedge, cedge_scaled, cv::Size(gray.cols * scaleFactor, gray.rows * scaleFactor));
            
            runThroughGraph(start, end, grayed_scaled, cedge_scaled, ImagePath_scaled, ImagePathEdge_scaled);
            // reconstructing the image to the original size for printing
            
            resize(ImagePath_scaled, ImagePath, Size(ImagePath_scaled.cols*scaleFactorBack, ImagePath_scaled.rows*scaleFactorBack));
            resize(ImagePathEdge_scaled, ImagePathEdge, Size(ImagePath_scaled.cols * scaleFactorBack, ImagePath_scaled.rows * scaleFactorBack));
        }
        else
        {
            processImageIntoCannyEdges(showing, image, start, end, gray, cedge);
            runThroughGraph(start, end, gray, cedge, ImagePath, ImagePathEdge);
        }

        //If the show boolean is true, then the function will show the images in new windows.
        if (showing = true)
        {
            print_images("original", image);
            print_images("gray", gray);
            print_images("Canny Black/White", cedge);
            
            if (resizing)
            {
                print_images("gray scaled", grayed_scaled);
                print_images("Canny scaled", cedge_scaled);
                print_images("Image with Path: ", ImagePath_scaled);
                print_images("Image with Path edge: ", ImagePathEdge_scaled);
            }

            print_images("Image with Path: ", ImagePath);
            print_images("Image with Path edge: ", ImagePathEdge);
            waitKey(0);
            destroyAllWindows();
        }

    }


    //else: print an error message
    else
    {
        printf("Incorrect inputs.... start and end points are inconsistent   \n");
    }


    return 0;
}
