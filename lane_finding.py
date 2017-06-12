# In[0]
#Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


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

        # Choose offset from image corners to plot detected cornersDrawnOn
        # offset = 100 # offset for dst points
        # # Grab image shape
        # img_size = (gray.shape[1], gray.shape[0])
        #
        # # define 4 source points
        # src = np.float32([corners[0], corners[9 -1], corners[-1], corners[-9]])
        #
        # # define 4 destination


# In[3]
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






# In[4]

# Combine Color and gradient thresholds for lane detection

image = mpimg.imread('test_images/test5.jpg')

def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] =1
    return combined_binary

result = threshold(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result, cmap = plt.get_cmap('gray'))
ax2.set_title('Combined S channel and gradient thresholds', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('output_images/combined_binary.jpg')


# In[5]

# Perspective Transform

# read in and display original image
img = mpimg.imread('test_images/straight_lines1.jpg')
plt.imshow(img)
plt.show()

# Source image points
imshape = img.shape
plt.imshow(img)
plt.plot(700, 443, '.') # top right
plt.plot(1200, imshape[0], '.') # bottom right
plt.plot(580, 443, '.') # top left
plt.plot(150, imshape[0], '.') # bottom left

# In[6]
# Define perspective transform function

# Four source coordinates
src = np.float32(
    [[700, 450],
     [1150, imshape[0]],
     [580, 443],
     [164, imshape[0]]])

# Four desired coordinates
dst = np.float32(
    [[975, (imshape[0] - imshape[0])],
     [975, imshape[0]],
     [300, (imshape[0] - imshape[0])],
     [300, imshape[0]]])

# def warp(img, src, dst):
#
#     # Compute and apply perpective transform
#     img_size = (img.shape[1], img.shape[0])
#     M = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
#
#     return warped

def warp(img, src, dst):
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    img_size = (img.shape[1], img.shape[0])

    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)

    # Could compute the inverse also by swapping the input parameeters
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create warped image - uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped



# In[7]
# Get perspective transform
warped_im = warp(img, src, dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

ax1.set_title('Source image')
ax1.imshow(img)
ax2.set_title('Warped image')
ax2.imshow(warped_im)
plt.savefig('output_images/perspectiveTransform.jpg')



# In[8]
# Test all steps so far
img = mpimg.imread('test_images/straight_lines1.jpg')
# plt.imshow(img)

# undistort with camera calibration
undistorted = cal_undistort(img, objpoints, imgpoints)
# plt.imshow(undistorted)

# Color/Gradient Thresholding
threshold_img = threshold(undistorted)
# plt.imshow(threshold_img, cmap = plt.get_cmap('gray'))

# perspective transform
warped_im = warp(threshold_img, src, dst)
plt.imshow(warped_im, cmap = plt.get_cmap('gray'))

plt.savefig('output_images/undistort+threshold+perspectiveTransform.jpg')
