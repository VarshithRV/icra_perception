import cv2
import numpy as np
import pandas as pd
import os
import sys


################################ DATASET PARAMETERS AND CONSTANTS ################################
CHEST_CAM_OFFSET = 0.8 # meters, distance from the chest camera to the table
CHEST_CAM_FOCAL_LENGTH = 0.00304 # meters, focal length of the chest camera
DON_DEPTH = 0.045 # depth of the bowl
DON_RADIUS = 0.0575 # radius of the bowl
ETA  =  0.000005 # magic number
CENTER_X = 240 # center of the DON 640
CENTER_Y = 320 # center of the DON 480
###################################################################################################


def value(x,y,center_x, center_y):
    d = CHEST_CAM_OFFSET - ETA*((x-center_x)**2 + (y-center_y)**2)
    d = np.random.normal(d, 0.001)
    if d >= CHEST_CAM_OFFSET - DON_DEPTH:
        return d
    else : 
        return CHEST_CAM_OFFSET

# generate an image of resolution 640x480 with a white background with 0s in all cells
image = np.zeros((480,640))

# generate the depth image
for x in range(0, 640):
    for y in range(0, 480):
        image[y][x] = value(x,y,CENTER_X,CENTER_Y)

print(image.min(), image.max())
# convert the image matrix to a 8-bit image, darker => deeper
image_prime = 1-(image - np.min(image))/(np.max(image) - np.min(image))
image_prime = image_prime*255
image_prime = image_prime.astype(np.uint8)

# show the image
cv2.imshow('image', image_prime)
cv2.waitKey(0)

# generate mask for the boundary of the DON
min_depth = np.min(image)
max_depth = np.max(image)
threshold = 0.0001

# generate the mask
mask = np.zeros((480,640))
for x in range(0, 640):
    for y in range(0, 480):
        if image[y][x] < CHEST_CAM_OFFSET - (DON_DEPTH - threshold):
            mask[y][x] = 1

# convert the mask matrix to a 8-bit image
mask = mask*255
mask = mask.astype(np.uint8)

# show the mask
cv2.imshow('mask', mask)
cv2.waitKey(0)

# find the center of the DON given its mask
M = cv2.moments(mask)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print(cX, cY)




