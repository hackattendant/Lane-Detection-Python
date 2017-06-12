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
img = mpimg.imread('camera_cal/calibration1.jpg')
plt.imshow(img)


# In[2]
# Arrays to store object points and image points from all the images

objpoints = [] # #D points in real world space
imgpoints = [] # 2D points in image plane
