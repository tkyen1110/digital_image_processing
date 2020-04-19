import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread('pasta.jpeg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur the image
gauss = cv2.GaussianBlur(img, (7,7), 0)
# Apply Unsharp masking
k = 1
unsharp_image = cv2.addWeighted(img, k+1, gauss, -k, 0)

plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(unsharp_image)
plt.show()