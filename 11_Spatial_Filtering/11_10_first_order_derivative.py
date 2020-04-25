import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a image with black background and white circle
img = np.zeros((500,500),dtype='uint8') 
cv2.circle(img,(250,250), 150, (255,255,255), -1)
 
# Output dtype = cv2.CV_8U
sob_8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
 
# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobel_64 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
abs_64 = np.absolute(sobel_64)
sobel_8u = np.uint8(abs_64)

plt.subplot(1,3,1)
plt.title("input image")
plt.imshow(img, cmap ='gray')

plt.subplot(1,3,2)
plt.title("sob_8u")
plt.imshow(sob_8u, cmap ='gray')

plt.subplot(1,3,3)
plt.title("sobel_8u(64F)")
plt.imshow(sobel_8u, cmap ='gray')

plt.show()