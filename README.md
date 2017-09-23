# self-driving-car-project-4-advanced-lane-finding
Repository for Udacity's Self-Driving Car Project 4: Advanced Lane Finding

Project Overview
---
In the Advanced Lane Finding Project, the following goals will be met:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Create a thresholded binary image through the use of HLS color transforms, image gradients, and other image processing techniques
* Apply a perspective transform to create a "birds-eye view" of the binary image
* Apply a distortion correction to raw images
* Detect lane pixels and fit the pixels to find the lane boundary
* Determine the curvature of the lane and vehicle position w.r.t. center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

[image1]: ./sample_images/distort.jpg "Distorted"
[image2]: ./sample_images/undistort.jpg "Undistorted"

Included Files
---
* README.me (this file): Contains my writeup, design, and implementation discussions
* lanedetect.py: Python code for performing camera calibration and image pipeline
* matrices.P: Pickled file containing camera calibration matrix and distortion coefficients
* sample_images/: This folder contains the output images for this pipeline    

Camera Calibration
---
The code for my implementation of camera calibration is located on lines <start number> to <end number> in lanedetect.py.

To perform camera calibration, I first prepare a list of "object points".  These object points are the (X, Y, Z) coordinates of the chessboards' corners.  Since the chessboard is on a flat image plan, I will be holding the Z-coordinate constant at Z = 0.  The origin of the chessboard corners is the upper-left corner; this will be (0,0,0).  The right-most corner will have object point (8, 5, 0), as there are 9 x 6 chessboard corners and we'll be using 0-indexing.

For each of the calibration images, we're looking for 9x6 = 54 corners on each image.  After running OpenCV's findChessboardCorners() function, if all 54 corners were found, I will append the list of chessboard corners in image coordinates to the "imgPoints" list and a copy of the object points "objP" to the "objPoints" list.  Once these lists are populated, it's a simple matter of using OpenCV's calibrateCamera() function to compute the distortion coefficients and camera calibration matrix.

An example of applying the computed matrix/coefficients is as follows.  Using these parameters with OpenCV's undistort() function, we turn the original distorted image:

![Distorted Image][image1]

into an undistorted image:

![Undistorted Image][image2]              
