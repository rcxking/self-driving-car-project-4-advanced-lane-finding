'''
lanedetect.py

Main function to perform advanced lane finding for Udacity Project 4.

Bryant Pong
9/19/17
'''

import sys
import pickle
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Image and Video Folders:
CAMERA_CAL = "./camera_cal/"
PICKLED_MATRICES = "./matrices.p"

'''
This function performs camera calibration
on a set of images located in the folder
denoted by CAMERA_CAL.  The distortion and
camera matrices will be stored in a pickled
object file.
'''
def CalibrateCamera():
    print( "Now calibrating camera" ) 

    # The calibration images are 9 corners wide
    # by 6 corners high.
    nx = 9
    ny = 6

    # These arrays hold the image and object points.
    # Image points are the points in 2D space, while object
    # points are the points in 3D real-world space.  Since the chessboard is 
    # against a flat surface, I'll be holding the Z coordinate constant at 0.
    #
    # The origin (0,0,0) in object coordinates is the upper-left corner.
    imgPoints = []
    objPoints = []

    # Initialize the object points to a 3-tuple (x, y, z):
    objP = np.zeros( ( nx * ny, 3 ), np.float32 )
    objP[ :, :2 ] = np.mgrid[ 0:nx,0:ny].T.reshape( -1 ,2 )

    # This glob makes it easier to open the calibration
    # images.  Calibration images are all labeled as "calibration*.jpg"
    calibrationImages = glob.glob( CAMERA_CAL + "calibration*.jpg" )

    # Iterate through all the images and find chessboard corners: 
    for image in calibrationImages:
        
        # Open the next image in BGR format:
        nextImg = cv2.imread( image )

        # Convert the image to grayscale:
        gray = cv2.cvtColor( nextImg, cv2.COLOR_BGR2GRAY )
   
        # Attempt to find the chessboard corners:
        ret, corners = cv2.findChessboardCorners( gray, ( nx, ny ), None )

        if ret:
            # We've found all the chessboard corners for this image.  Append
            # the corners to the imgPoints array and the object points to the
            # objPoints array:
            imgPoints.append( corners )
            objPoints.append( objP )

            '''
            cv2.drawChessboardCorners( nextImg, ( nx, ny ), corners, ret )
            plt.imshow( nextImg )
            plt.show()
            '''

    # Now that we have our image and object points, we can proceed
    # to calibrating the camera:

    # The shape of an image.  This is found by printing out the shape of the grayscale images:
    imgShape = ( 1280, 720 )
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objPoints, imgPoints, imgShape, None, None )

    # Save the distortion and camera matrices to a pickled file:
    matrices = { "mtx":mtx, "dist":dist }
    with open( PICKLED_MATRICES, "wb" ) as f:
        pickle.dump( matrices, f )

    print( "Done calibrating camera" )

# Main function.  We'll be accepting one command line argument, which allows
# the user to specify whether they want to calibrate the camera, or load
# the saved camera and distortion matrices from a pickled file.
def main():
    
    if len( sys.argv ) > 1:
        # Check if we'll be performing camera calibration:
        if "calibrate" in sys.argv:
            print( "Will be performing camera calibration" )
            CalibrateCamera()
    else:
        # We'll be loading the matrices from file:
        print( "Will be loading camera and distortion matrices from file" )
        with open( PICKLED_MATRICES, "rb" ) as f:
            pickledMatrix = pickle.load( f )

        # Load the distortion and camera matrices:
        dist = pickledMatrix[ "dist" ]
        mtx = pickledMatrix[ "mtx" ]


# Main function runner:
if __name__ == "__main__":
    main()
