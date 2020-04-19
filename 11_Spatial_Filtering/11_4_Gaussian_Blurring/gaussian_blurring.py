import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('../opencv_logo.png', cv2.IMREAD_COLOR)

# Creates a 1-D Gaussian kernel
a = cv2.getGaussianKernel(5,1)

# Apply the above Gaussian kernel. Here, I
# have used the same kernel for both X and Y
b = cv2.sepFilter2D(img,-1,a,a)

# Apply the Gaussian blur
c = cv2.GaussianBlur(img,(5,5),1)

plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(b)
plt.subplot(2,2,3)
plt.imshow(c)
plt.show()