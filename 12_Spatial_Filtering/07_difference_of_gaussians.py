import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('opencv_logo.png', cv2.IMREAD_GRAYSCALE)

# Apply 3x3 and 7x7 Gaussian blur
low_sigma = cv2.GaussianBlur(img,(3,3),0)
high_sigma = cv2.GaussianBlur(img,(5,5),0)

# Calculate the DoG by subtracting
dog = low_sigma - high_sigma

plt.subplot(1,2,1)
plt.imshow(img, cmap ='gray')
plt.subplot(1,2,2)
plt.imshow(dog, cmap ='gray')
plt.show()