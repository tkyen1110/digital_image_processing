import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image in the greyscale
img = cv2.imread('adap1.png', cv2.IMREAD_GRAYSCALE)

# Apply Otsu method
ret, thres = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Apply adaptive threshold
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

blur = cv2.GaussianBlur(img, (5,5), 1)
row, column = np.where(img > blur-2)
blur[row, column] = 255

plt.subplot(2,2,1)
plt.title("original")
plt.imshow(img, cmap ='gray')

plt.subplot(2,2,2)
plt.title("Otsu method")
plt.imshow(thres, cmap ='gray')

plt.subplot(2,2,3)
plt.title("cv2 adaptive threshold")
plt.imshow(th3, cmap ='gray')

plt.subplot(2,2,4)
plt.title("adaptive threshold")
plt.imshow(blur, cmap ='gray')

plt.show()