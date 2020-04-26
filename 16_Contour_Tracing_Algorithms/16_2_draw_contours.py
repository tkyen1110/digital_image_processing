import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('opencv_logo.png')
img_contours = img.copy()
# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold the image to produce a binary image
ret, thresh = cv2.threshold(img_gray, 150, 255, 0)

# Find the contours
contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw the contours
cv2.drawContours(img_contours, contours, -1, (153,153,0), 3)

plt.subplot(2,2,1)
plt.title('original')
plt.imshow(img)

plt.subplot(2,2,2)
plt.title('grayscale')
plt.imshow(img_gray, cmap ='gray')

plt.subplot(2,2,3)
plt.title('binary image')
plt.imshow(thresh, cmap ='gray')

plt.subplot(2,2,4)
plt.title('contours')
plt.imshow(img_contours)

plt.show()