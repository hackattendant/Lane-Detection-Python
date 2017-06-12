# In[0]
#Imports

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
%matplotlib inline

# In[1]
# Compute the camera calibration matrix and distortion coefficents

# Read in and make a list of calibration images
images = glob. glob('camera_cal/calibration*.jpg')

# Read in a calibration image
img = mpimg.imread('camera_cal/calibration2.jpg')
plt.imshow(img)


# In[2]
# Arrays to store object points and image points from all the images

objpoints = [] # #D points in real world space
imgpoints = [] # 2D points in image plane


# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ....., (7,5,0) for points on chessboard
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

# Iterate through each image file, detecting corenrs, and appending points to the obj and img Arrays
for fname in images:
    # Read in each image
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.imshow(img)
        plt.savefig('output_images/draw_corners/' + str(fname))
        plt.savefig('output_images/cornersDrawnOn.jpg')


# In[2]
# Undistort image
def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

distorted_img = mpimg.imread('camera_cal/calibration1.jpg')

undistorted = cal_undistort(distorted_img, objpoints, imgpoints)

# Visualize results of undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(distorted_img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('output_images/undistortedImage.jpg')
