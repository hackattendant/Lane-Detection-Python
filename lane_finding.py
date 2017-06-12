# In[0]
#Imports

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# In[1]
# Compute the camera calibration matrix and distortion coefficents

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

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    #draw and display the corners
    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    plt.imshow(img)
    plt.savefig('drawChessboardCorners.jpg')
