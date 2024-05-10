import cv2
import numpy as np
import pandas as pd
import os
import sys
import time

################################ DATASET PARAMETERS AND CONSTANTS ################################
CHEST_CAM_OFFSET = 0.8 # meters, distance from the chest camera to the table
CHEST_CAM_FOCAL_LENGTH = 0.00304 # meters, focal length of the chest camera
DON_DEPTH = 0.045 # depth of the bowl
DON_RADIUS = 0.0575 # radius of the bowl
ETA  =  0.000008 # magic number
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
    

image = np.zeros((480,640))

# generate the depth image
for x in range(0, 640):
    for y in range(0, 480):
        image[y][x] = value(x,y,CENTER_X,120)

image2 = np.zeros((480,640))

# generate the depth image
for x in range(0, 640):
    for y in range(0, 480):
        image2[y][x] = value(x,y,CENTER_X+100,320)

image = image + image2
image -= CHEST_CAM_OFFSET


# convert the image matrix to a 8-bit image, darker => deeper
image_prime = 1-(image - np.min(image))/(np.max(image) - np.min(image))
image_prime = image_prime*255
image_prime = image_prime.astype(np.uint8)

# show the image
cv2.imshow('image', image_prime)
# cv2.waitKey(0)
# save the image
cv2.imwrite('don.png', image_prime)

#####################################perception starts here#####################################
duration = time.time()

# generate mask for the boundary of the DON
centers = {}

min_depth = np.min(image)
max_depth = np.max(image)
threshold = 0.0001

# generate the mask
mask1 = np.zeros((480,640))
mask2 = np.zeros((480,640))
for x in range(0, 640):
    for y in range(0, 480):
        if image[y][x] < CHEST_CAM_OFFSET - (DON_DEPTH - threshold):
            if y < 240:
                mask1[y][x] = 1
            else:
                mask2[y][x] = 1

mask1 = mask1*255
mask1 = mask1.astype(np.uint8)
mask2 = mask2*255
mask2 = mask2.astype(np.uint8)

M = cv2.moments(mask1)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
centers['mask1'] = (cX, cY)

M = cv2.moments(mask2)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(image_prime, (cX, cY), 5, (255, 0, 0), -1)
centers['mask2'] = (cX, cY)

print(centers)

duration = time.time() - duration
print("One Iteration of perception in secods : ",duration)
print("X error : ", (((centers['mask1'][0] - CENTER_X)/CENTER_X + (centers['mask2'][0] - (CENTER_X+100))/(CENTER_X+100) ) *100)/ 2)
print("Y error : ", (((centers['mask1'][1] - 120)/120 + (centers['mask2'][1] - 320)/320 )*100)/ 2)