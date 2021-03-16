import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('cars.jpg')
img_contours = img.copy()
img_convex_hull = img.copy()
# Convert it to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold the image
ret, thresh = cv2.threshold(img_gray,50,255,0)
# Find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw the contours
cv2.drawContours(img_contours, contours, -1, (255,0,0), 2)

# For each contour, find the convex hull and draw it
# on the original image.
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    cv2.drawContours(img_convex_hull, [hull], -1, (255, 0, 0), 2)

plt.subplot(2,3,1)
plt.title('original')
plt.imshow(img)

plt.subplot(2,3,2)
plt.title('grayscale')
plt.imshow(img_gray, cmap ='gray')

plt.subplot(2,3,3)
plt.title('binary image')
plt.imshow(thresh, cmap ='gray')

plt.subplot(2,3,4)
plt.title('contours')
plt.imshow(img_contours)

plt.subplot(2,3,5)
plt.title('convex hull')
plt.imshow(img_convex_hull)

plt.show()