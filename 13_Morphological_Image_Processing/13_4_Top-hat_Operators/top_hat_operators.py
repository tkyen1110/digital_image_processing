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

# Structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
# Apply the opening operation
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# Apply the closing operation
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# Apply the top hat transform
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# Apply the black hat transform
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

plt.subplot(2,4,1)
plt.title("original")
plt.imshow(img, cmap ='gray')

plt.subplot(2,4,2)
plt.title("Otsu method")
plt.imshow(thres, cmap ='gray')

plt.subplot(2,4,3)
plt.title("cv2 adaptive threshold")
plt.imshow(th3, cmap ='gray')

plt.subplot(2,4,4)
plt.title("adaptive threshold")
plt.imshow(blur, cmap ='gray')

plt.subplot(2,4,5)
plt.title("opening")
plt.imshow(opening, cmap ='gray')

plt.subplot(2,4,6)
plt.title("closing")
plt.imshow(closing, cmap ='gray')

plt.subplot(2,4,7)
plt.title("tophat")
plt.imshow(tophat, cmap ='gray')

plt.subplot(2,4,8)
plt.title("blackhat")
plt.imshow(blackhat, cmap ='gray')

plt.show()