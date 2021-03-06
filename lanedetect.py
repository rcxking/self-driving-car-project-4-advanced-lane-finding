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
from moviepy.editor import VideoFileClip

# Image and Video Folders:
CAMERA_CAL = "./camera_cal/"
TEST_IMAGES = "./test_images/"
PICKLED_MATRICES = "./matrices.p"

# Set this flag to True to display images:
displayImages = False

'''
Helper function to display an image with
PyPlot:
'''
def DisplayImage( img, overrideDisplayImages = False ):

    if not displayImages and not overrideDisplayImages:
        return

    plt.imshow( img )
    plt.show()

'''
Helper function to display a grayscale image
with Pyplot:
'''
def DisplayGrayImage( img ):

    if not displayImages:
        return

    plt.imshow( img, cmap='gray' )
    plt.show()

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

'''
The following functions perform thresholding on the L-Channel
in LUV color space and on the B-Channel in LAB color space.

The L-Channel thresholding is used to help identify white lane
lines while the B-Channel thresholding is used to help identify
yellow lane lines.
'''
def ThresholdLChannel( img, threshMin=0, threshMax=255 ):

    # Convert the BGR image to LUV Color Space:
    luv = cv2.cvtColor( img, cv2.COLOR_BGR2LUV )

    # Acquire the L-Channel and apply the thresholding:
    l = luv[ :, :, 0 ]
    binaryOutput = np.zeros_like( l )
    binaryOutput[ ( l > threshMin ) & ( l <= threshMax ) ]  = 1
    
    return binaryOutput

def ThresholdBChannel( img, threshMin=0, threshMax=255 ):

    # Convert the BGR image to LAB Color Space:
    lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB )

    # Acquire the B-Channel and apply the thresholding:
    b = lab[ :, :, 2 ]
    binaryOutput = np.zeros_like( b )
    binaryOutput[ ( b > threshMin ) & ( b <= threshMax ) ] = 1

    return binaryOutput

'''
Helper function to apply a Region of Interest (ROI)
to the given image.
'''
def RegionOfInterest( img, vertices ):

    mask = np.zeros_like( img )

    # Determine if this is a 3 channel (RGB) or 1 channel (grayscale) image:
    if len( img.shape ) > 2:
        # 3 Channel Image:
        channelCount = img.shape[ 2 ]
        ignoreMaskColor = ( 255, ) * channelCount
    else:
        # 1 Channel Image
        ignoreMaskColor = 255

    # Fill pixels inside the ROI defined by vertices:
    cv2.fillPoly( mask, vertices, ignoreMaskColor )

    # Return the masked image:
    maskedImage = cv2.bitwise_and( img, mask )

    return maskedImage

'''
Image Pipeline.  Given an input image, the camera
calibration matrix, and the distortion coefficients,
find and mark the lane lines.  Return the image with the
lane lines marked.

IMPORTANT: This function expects an image in RGB color space.
The output of this function will also be an image in RGB color space.
'''
def ImagePipeline( img ):

    # Load the camera calibration matrix and distortion coefficients:
    with open( PICKLED_MATRICES, "rb" ) as f:
        pickledMatrix = pickle.load( f )
    dist = pickledMatrix[ "dist" ]
    mtx = pickledMatrix[ "mtx" ]

    img = cv2.cvtColor( img, cv2.COLOR_RGB2BGR )

    # First, undistort the image:
    undistort = cv2.undistort( img, mtx, dist, None, mtx )
    cvtImage = cv2.cvtColor( undistort, cv2.COLOR_BGR2RGB )
    DisplayImage( cvtImage )

    # Now perform thresholding to eliminate noise and to extract
    # the lane lines.

    # Perform Color Thresholding on the L-Channel in LUV color space.
    # This allows better identifying of white lines:
    lThresh = ThresholdLChannel( undistort, 215, 255 )
    DisplayGrayImage( lThresh )

    # Perform Color Thresholding on the B-Channel in LAB color space.
    # This allows better identifying of yellow lines:
    bThresh = ThresholdBChannel( undistort, 145, 255 )
    DisplayGrayImage( bThresh )

    # Now combine the different thresholds into 1 image:
    combined = np.zeros_like( lThresh )
    combined[ ( ( lThresh == 1 ) | ( bThresh == 1 ) ) ] = 1

    # DEBUG ONLY: Display the combined binary image:
    DisplayGrayImage( combined ) 

    # Apply a ROI mask to trim areas where the lane lines aren't:
    roiVertices = np.array( [ [ ( 186, 704 ), ( 615, 427 ), ( 700, 427 ), ( 1150, 704 ) ] ], dtype=np.int32 )

    combined = RegionOfInterest( combined, roiVertices )
    DisplayGrayImage( combined )

    # Now apply a Perspective Transformation to the image to make the lines
    # appear as parallel.  Points picked via the GIMP image program. 
    srcPoints = np.float32( [ [ 230, 703  ], [ 581, 460 ], [ 700, 460 ], [ 1070, 703 ] ] )
    dstPoints = np.float32( [ [ 375, 720 ], [ 375, 0   ], [ 1000,  0 ], [ 1000, 720 ] ] )

    M = cv2.getPerspectiveTransform( srcPoints, dstPoints )
    warped = cv2.warpPerspective( combined, M, ( combined.shape[ 1 ], combined.shape[ 0 ] ), flags = cv2.INTER_LINEAR )

    # DEBUG ONLY: Display the warped image:
    DisplayGrayImage( warped )

    # Acquire the lane line computations (including the curvature and offset in meters) 
    laneLines, left_fitx, right_fitx, ploty, avgCurveRad, offsetInMeters = MarkLaneLines( warped )

    # Draw the lane in the original image:
    grayUndistort = cv2.cvtColor( undistort, cv2.COLOR_BGR2GRAY )
    warpZero = np.zeros_like( grayUndistort ).astype( np.uint8 )
    colorWarp = np.dstack( ( warpZero, warpZero, warpZero ) )

    pts_left = np.array( [ np.transpose( np.vstack( [ left_fitx, ploty ] ) ) ] )
    pts_right = np.array( [ np.flipud( np.transpose( np.vstack( [ right_fitx, ploty ] ))) ] )
    pts = np.hstack( ( pts_left, pts_right ) )
    cv2.fillPoly( colorWarp, np.int_([ pts ] ), ( 0, 255, 0 ) )

    # Compute the inverse perspective transform and apply the lane drawing back to the original image 
    Minv = cv2.getPerspectiveTransform( dstPoints, srcPoints )
    newWarp = cv2.warpPerspective( colorWarp, Minv, ( img.shape [ 1 ], img.shape[ 0 ] ) )

    result = cv2.addWeighted( undistort, 1, newWarp, 0.3, 0 )

    # Display the Radius and Position Information:
    radiiText = "Radius of Curvature = " + str( avgCurveRad ) + "(m)"
    cv2.putText( result, radiiText, ( 0, 50 ), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255 ), 2, cv2.LINE_AA )

    posText = "Vehicle is " + str( round( abs( offsetInMeters ), 2 ) ) + "m"

    # If the offset is positive, the lane is to the right
    # of the center, so we're drifting left:
    if offsetInMeters > 0.0:
        posText += " left of center"
    else:
        posText += " right of center"

    cv2.putText( result, posText, ( 0, 100 ), cv2.FONT_HERSHEY_SIMPLEX, 2, ( 255, 255, 255 ), 2, cv2.LINE_AA )

    # Convert the image back to RGB color space:
    result = cv2.cvtColor( result, cv2.COLOR_BGR2RGB ) 
    DisplayImage( result )


    return result

'''
This function takes in a warped binary image and finds/marks
the lane lines in the image: 
'''
def MarkLaneLines( warped ): 

    # Calculate a Histogram of the lower half of the warped image:
    hist = np.sum( warped[ warped.shape[ 0 ] // 2:,: ], axis = 0 )

    # Output image to see the lane lines:
    outputImage = np.dstack( ( warped, warped, warped ) ) * 255

    # Find the left and right peaks of the histogram; these are
    # respectively the left and right lane lines:
    midpoint = np.int( hist.shape[ 0 ] / 2 )
    leftx_base = np.argmax( hist[ :midpoint] )
    rightx_base = np.argmax( hist[ midpoint: ] ) + midpoint

    # Number of sliding windows:
    numWindows = 9

    # Height of the windows:
    windowHeight = np.int( warped.shape[ 0 ] / numWindows )

    # Find the X and Y Positions of all non-zero pixels
    nonZero = warped.nonzero()
    nonZeroY = np.array( nonZero[ 0 ] )
    nonZeroX = np.array( nonZero[ 1 ] )

    # Current Position of the window:
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Margin for each window:
    margin = 100

    # Minumum number of pixels found to recenter the window:
    minpix = 10

    # These lists hold the pixels for the left and right lanes:
    left_lane_inds = []
    right_lane_inds = []

    # Sliding Window:
    for window in range( numWindows ):

        # Calculate Window Boundaries:
        win_y_low = warped.shape[ 0 ] - ( window + 1 ) * windowHeight
        win_y_high = warped.shape[ 0 ] - window * windowHeight
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the output image:
        cv2.rectangle( outputImage, ( win_xleft_low, win_y_low ), ( win_xleft_high, win_y_high ), ( 0, 255, 0 ), 2 )
        cv2.rectangle( outputImage, ( win_xright_low, win_y_low ), ( win_xright_high, win_y_high ), ( 0, 255, 0 ), 2 )

        # Find the non-zero pixels in the window:
        good_left_inds = ( ( nonZeroY >= win_y_low ) & ( nonZeroY < win_y_high ) & ( nonZeroX >= win_xleft_low ) & ( nonZeroX < win_xleft_high ) ).nonzero()[ 0 ]
        good_right_inds = ( ( nonZeroY >= win_y_low ) & ( nonZeroY < win_y_high ) & ( nonZeroX >= win_xright_low ) & ( nonZeroX < win_xright_high ) ).nonzero()[ 0 ]

        # Append these points to the lists:
        left_lane_inds.append( good_left_inds )
        right_lane_inds.append( good_right_inds )

        # Recenter the windows as needed:
        if len( good_left_inds ) > minpix:
            leftx_current = np.int( np.mean( nonZeroX[ good_left_inds ] ) )
        if len( good_right_inds ) > minpix:
            rightx_current = np.int( np.mean( nonZeroX[ good_right_inds ] ) )

    # Concatenate the lists of indices:
    left_lane_inds = np.concatenate( left_lane_inds )
    right_lane_inds = np.concatenate( right_lane_inds )

    # Extract left and right lane pixel positions:
    leftX = nonZeroX[ left_lane_inds ]
    leftY = nonZeroY[ left_lane_inds ]
    rightX = nonZeroX[ right_lane_inds ]
    rightY = nonZeroY[ right_lane_inds ]

    '''
    print( len( leftX ) )
    print( len( leftY ) )
    print( len( rightX ) )
    print( len( rightY ) )
    '''

    #if leftX or leftY or rightX 

    # Fit a 2nd Order Polynomial:
    leftFit = np.polyfit( leftY, leftX, 2 )
    rightFit = np.polyfit( rightY, rightX, 2 )

    ploty = np.linspace( 0, warped.shape[ 0 ] - 1, warped.shape[ 0 ] )
    left_fitx = leftFit[ 0 ] * ploty ** 2 + leftFit[ 1 ] * ploty + leftFit[ 2 ]
    right_fitx = rightFit[ 0 ] * ploty ** 2 + rightFit[ 1 ] * ploty + rightFit[ 2 ]

    # Compute the lane curvature:
    ym_per_pix = 30 / 720 # Meters per pixel in Y
    xm_per_pix = 3.7 / 700 # Meters per pixel in X

    y_eval = np.max( ploty )

    # Convert from pixel measurements to meters:
    left_fit_cr = np.polyfit( leftY * ym_per_pix, leftX * xm_per_pix, 2 )
    right_fit_cr = np.polyfit( rightY * ym_per_pix, rightX * xm_per_pix, 2 )

    left_curverad = ( ( 1 + ( 2 * left_fit_cr[ 0 ] * y_eval * ym_per_pix + left_fit_cr[ 1 ] ) ** 2 ) ** 1.5 ) / np.absolute( 2 * left_fit_cr[ 0 ] )

    right_curverad = ( ( 1 + ( 2 * right_fit_cr[ 0 ] * y_eval * ym_per_pix + right_fit_cr[ 1 ] ) ** 2 ) ** 1.5 ) / np.absolute( 2 * right_fit_cr[ 0 ] )

    print( left_curverad, 'm', right_curverad, 'm' )

    # Average the turning radii:
    avgCurveRad = int( ( left_curverad + right_curverad ) / 2 )
   
    '''
    Compute the position offset of the car.  To compute the offset, we'll look
    for the bottom-most positions of the left and right lane and average them.    
    '''
    bottomLeftX = leftX[ leftY.argmax() ]
    bottomRightX = rightX[ rightY.argmax() ]
    offsetInPixels = ( ( bottomLeftX + bottomRightX ) / 2 ) - 640
    offsetInMeters = offsetInPixels * xm_per_pix

    # Enable this next section if you want to see sliding windows and lane fitting:
    if displayImages:
        outputImage[ nonZeroY[ left_lane_inds ], nonZeroX[ left_lane_inds ] ] = [ 255, 0, 0 ]
        outputImage[ nonZeroY[ right_lane_inds ], nonZeroX[ right_lane_inds ] ] = [ 0, 0, 255 ]
    
        plt.imshow( outputImage )
        plt.plot( left_fitx, ploty, color = 'yellow' )
        plt.plot( right_fitx, ploty, color = 'yellow' )
        plt.xlim( 0, 1280 )
        plt.ylim( 720, 0 )
        plt.show() 

    return outputImage, left_fitx, right_fitx, ploty, avgCurveRad, offsetInMeters 


# Main function.  We'll be accepting one command line argument, which allows
# the user to specify whether they want to calibrate the camera, or load
# the saved camera and distortion matrices from a pickled file.
def main():
    
    if len( sys.argv ) > 1:
        # Check if we'll be performing camera calibration:
        if "calibrate" in sys.argv:
            print( "Will be performing camera calibration" )
            CalibrateCamera()

        # Run the image pipeline on the images in the TEST_IMAGES folder:
        if "images" in sys.argv:
            print( "Now running image pipeline on images in " + TEST_IMAGES )

            images = glob.glob( TEST_IMAGES + "*.jpg" )
            images.sort() 
            for imageName in images:

                print( "Processing " + imageName )

                # Open the image in BGR color space: 
                nextImage = cv2.imread( imageName ) 

                # The image pipeline is expecting an RGB image:
                nextImage = cv2.cvtColor( nextImage, cv2.COLOR_BGR2RGB ) 

                laneLines = ImagePipeline( nextImage )  

                DisplayImage( laneLines, True )

        # Run the image pipeline on the specified video (found in the same directory as this script )
        if "video" in sys.argv:

            # Check to see if we have a specified video file:
            videoIndex = sys.argv.index( "video" )
            if videoIndex + 1 >= len( sys.argv ):
                print( "ERROR: Expecting Video File" )
                return

            videoName = sys.argv[ videoIndex + 1 ]
            print( "Now running image pipeline on video: " + videoName )
            clip = VideoFileClip( "./" + videoName )
            outputVideo = "output_video/output.mp4"
            vid = clip.fl_image( ImagePipeline ) 
            vid.write_videofile( outputVideo, audio = False )

            print( "Done creating video in output_video/" )

    else:
        # We need at least one argument:
        print( "Usage: " + sys.argv[ 0 ] + " <calibrate> <images> <video> <video name>" )
        return

# Main function runner:
if __name__ == "__main__":
    main()
