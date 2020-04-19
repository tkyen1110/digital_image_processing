import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('../opencv_logo.png', cv2.IMREAD_COLOR)

# Apply the Bilateral filter
blur = cv2.bilateralFilter(img,5,25,25)

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(blur)
plt.show()