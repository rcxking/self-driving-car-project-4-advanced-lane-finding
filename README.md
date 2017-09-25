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

[image1]: ./output_images/distort.jpg "Distorted"
[image2]: ./output_images/undistort.jpg "Undistorted"
[image3]: ./output_images/undistorted_lanes.jpg "Undistorted Lanes"
[image4]: ./output_images/binary_threshold.jpg "Binary Threshold"
[image5]: ./output_images/roi.jpg "Region of Interest"
[image6]: ./output_images/perspective.jpg "Perspective Transform"
[image7]: ./output_images/lane_line_fit.png "Lane Line Fitting"
[image8]: ./output_images/pipeline_output.jpg "Pipeline Output"

Included Files
---
* README.me (this file): Contains my writeup, design, and implementation discussions
* lanedetect.py: Python code for performing camera calibration and image pipeline
* matrices.P: Pickled file containing camera calibration matrix and distortion coefficients
* output_images/: This folder contains the output images for this pipeline
* output_video/output.mp4: This file contains the output of project_video.mp4 after running my pipeline on it
* camera_cal/: Contains the chessboard images used for camera calibration
* test_images/: Contains images used for testing the image pipeline

Running the Code
---
Please use

python lanedetect.py <options>

to run my code.  The supported options are:
* images: Runs pipeline on all images in the test_images/ folder
* calibrate: Performs camera calibration and stores the camera matrix/distortion coefficients in a pickled file matrices.p
* video <video name>: Generates a video of the pipeline running on <video name>.  Stored as "output.mp4" in output_video/

For example, to generate only the calibration data:
python lanedetect.py calibrate

To run on all images in test_images/:
python lanedetect.py images

To run on a MP4 file (ex: project_video.mp4)  in this directory:
python lanedetect.py video project_video.mp4

You can also stack commands together:
python lanedetect.py calibrate images video project_video.mp4

Camera Calibration
---
The code for my implementation of camera calibration is located on lines 47 to 103 in lanedetect.py (CalibrateCamera()).

To perform camera calibration, I first prepare a list of "object points".  These object points are the (X, Y, Z) coordinates of the chessboards' corners.  Since the chessboard is on a flat image plan, I will be holding the Z-coordinate constant at Z = 0.  The origin of the chessboard corners is the upper-left corner; this will be (0,0,0).  The right-most corner will have object point (8, 5, 0), as there are 9 x 6 chessboard corners and we'll be using 0-indexing.

For each of the calibration images, we're looking for 9x6 = 54 corners on each image.  After running OpenCV's findChessboardCorners() function, if all 54 corners were found, I will append the list of chessboard corners in image coordinates to the "imgPoints" list and a copy of the object points "objP" to the "objPoints" list.  Once these lists are populated, it's a simple matter of using OpenCV's calibrateCamera() function to compute the distortion coefficients and camera calibration matrix.

An example of applying the computed matrix/coefficients is as follows.  Using these parameters with OpenCV's undistort() function, we turn the original distorted image:

![Distorted Image][image1]

into an undistorted image:

![Undistorted Image][image2]

Image Pipeline (Single Images)
---

1. Distortion Correction

In lanedetect.py lines 248 to 254, I first load the distortion coefficients and the camera matrix from the pickled file "matrices.p".  I then undistort the input image via cv2.undistort() to produce the following undistorted image:

![Undistorted Lanes][image3]

2. Generating a Thresholded Binary Image

I originally wrote up 4 functions that could be used to help with thresholding.  These 4 functions could perform: 1) Sobel Thresholding; 2) Gradient Magnitude Thresholding; 3) Gradient Orientation Thresholding; and 4) S-Channel Thresholding (after converting the image from BGR color space to HLS color space).  After some trial-and-error, I determined that I could get good results from just performing S-Channel and Gradient Magnitude Thresholding (lines 260 and 263).  I originally also wanted to add in gradient orientation thresholding, but my output images resulted in too much noise.

Once these two thresholds are computed, I combine them into a single binary image using lines 266 and 267.

An example of the resulting image is as follows:

![Binary Threshold][image4]

After this step, I was also interested in the pixels that corresponded to the lane lines.  To filter out the remaining data, I created a Region of Interest in the shape of a trapezoid.  These points were picked by using the GIMP image program (lines 270 and 272).

After applying this ROI, the thresholded image is as follows:

![Region of Interest][image5]

3. Perspective Transformation

To acquire a birds-eye view of the ROI Binary Thresholded image, I needed to perform a perspective transformation.  To do so, I found 4 points in the shape of a trapezoid on the ROI image that I wanted to warp; these points were picked by using the GIMP program (line 277).  Since I wanted the lane lines to be warped in such a way that the lines appeared parallel, I picked 4 destination points in the shape of a rectangle (line 278).  These destination points were chosen such that the start and ending points of each lane mapped to the top and bottom of the warped image.  The perspective transform is executed on lines 280 and 281.

A warped image is as follows:

![Perspective Transform][image6]  

As one can see, the lane lines now appear parallel w.r.t. each other in the warped image.

4. Identifying Lane Lines

I wrote a function "MarkLaneLines()" from lines 321 to 445 to find the lane lines and perform a 2nd-order polynomial fit for each of the left and right lanes.  This function implements a sliding window algorithm to find which pixels belong to the lanes.  The first step is to compute a histogram of the lower half of the warped perspective image; the highest peaks correspond to the starting locations of the left and right lane lines.  With these initial values, I compute 9 sliding windows for each lane, determining how many pixels lie within the window and determining whether the centers of these windows need to be shifted per iteration (lines 361 to 387).  Once this function completes, two lists containing the points belonging to each line are found (lines 390 and 391), and I then compute a 2nd-order polynomial for each of the best-fit lines through the lane points (lines 400 to 405).

The resulting image (with sliding windows and best-fit lines drawn) is as follows:

![Lane Line Fitting][image7]

5. Computing Radius of Curvature and Position of Vehicle

Lines 408 to 421 show the computations needed to compute the radius of the curvature.  First, I am making the assumption that there are (30/720) meters per pixel in the Y-Direction and (3.7/700) meters per pixel in the X-Direction.  I will then convert the left and right (X, Y) points from pixels to meters.  Finally, the left and right curvature radii are calculated by performing another 2nd-ordered polynomial fit and then using the radii formula presented in the lectures (lines 417 and 419).

To compute the turning radii, I simply averaged the left and right curvature radii (line 424).

To compute the position offsets, I averaged the X positions of the left and right lane, then subtracted the X-center of 640 (since the image widths are 1280; 1280 / 2 = 640).  Line 430.

6. Overlaying Lane and Curvature Results

Finally, lines 287-312 are where the lane is plotted back and warped to the original image.  The resulting image is as follows:

![Pipeline Output][image8]       

Pipeline (Video)
---
My pipeline running on the "project_video.mp4" can be found in the output_video/ folder as "output.mp4".

Here's a link to [output.mp4](./output_video/output.mp4)

Discussion on Possible Problems/Issues
---
One issue I've noticed results from the perspective transform.  As mentioned above, I selected 4 points in the shape of a trapezoid, with the farther points corresponding to the top of the trapezoid.  After warping this trapezoidal ROI, I've noticed that there are more errors near the top of the trapezoid (as the pixels are smaller and ffarther away).  One way to correct for this is to pick a shorter trapezoidal region; however the trade-off is that less of the lane can be detected with this approach (which is more problematic when driving a car at higher speeds).

In terms of implementation, the pipeline may be slow if implemented for a real-time system (as in the case of an actual self-driving car).  However, this is more of an implementation problem; the pipeline may execute faster if C++ was used instead of Python for this project.

Another problem seen was in the sliding windows.  In some of the test images, one can see that (especially for the dotted lane lines) no pixels were detected in the top of the image (farther away from the camera).  While the examples I've seen correctly fit the polynomial to where those points would have been, if more pixels were seen (for example by relaxing the thresholding), the polynomial would be better fit.          
