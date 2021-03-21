import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Load an image
img = cv2.imread('opencv_logo.png', cv2.IMREAD_COLOR)

# 1. Averaging
blur = cv2.boxFilter(img,-1,(5,5), normalize = True)

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(blur)
plt.show()


# 2. Median Blurring
# Add salt and pepper noise to the image
noise_img = random_noise(img, mode="s&p",amount=0.3)
noise_img = np.array(255*noise_img, dtype = 'uint8')
 
# Apply median filter
median = cv2.medianBlur(noise_img,5)

plt.subplot(1,2,1)
plt.imshow(noise_img)
plt.subplot(1,2,2)
plt.imshow(median)
plt.show()