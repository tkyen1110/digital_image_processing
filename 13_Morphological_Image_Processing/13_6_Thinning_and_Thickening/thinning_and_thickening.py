import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an image with text on it
img = np.zeros((100,400),dtype='uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'TheAILearner',(5,70), font, 2,(255),5,cv2.LINE_AA)
img1 = img.copy()

# Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# Create an empty output image to hold values
thin = np.zeros(img.shape,dtype='uint8')

# Loop until erosion leads to an empty set
while (cv2.countNonZero(img1)!=0):
    print("cv2.countNonZero(img1) = ", cv2.countNonZero(img1))

    plt.subplot(2,3,1)
    plt.title("original")
    plt.imshow(img1, cmap ='gray')

    # Erosion
    erode = cv2.erode(img1,kernel)
    # Opening on eroded image
    opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
    # Subtract these two
    subset = erode - opening
    # Union of all previous sets
    thin = cv2.bitwise_or(subset,thin)
    # Set the eroded image for next iteration
    img1 = erode.copy()

    plt.subplot(2,3,2)
    plt.title("erode")
    plt.imshow(erode, cmap ='gray')

    plt.subplot(2,3,3)
    plt.title("opening")
    plt.imshow(opening, cmap ='gray')

    plt.subplot(2,3,4)
    plt.title("subset")
    plt.imshow(subset, cmap ='gray')

    plt.subplot(2,3,5)
    plt.title("thin")
    plt.imshow(thin, cmap ='gray')

    plt.subplot(2,3,6)
    plt.title("next original")
    plt.imshow(img1, cmap ='gray')

    plt.show()

plt.subplot(1,2,1)
plt.title("original")
plt.imshow(img, cmap ='gray')

plt.subplot(1,2,2)
plt.title("thin")
plt.imshow(thin, cmap ='gray')

plt.show()