import cv2
import numpy as np
import matplotlib.pyplot as plt

#roi is the object or region of object we need to find
roi = cv2.imread('messi_ground.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#target is the image we search in
target = cv2.imread('messi.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

use_cv2_calcBackProject = True
if use_cv2_calcBackProject:
    # calculating object histogram
    M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(M,M,0,255,cv2.NORM_MINMAX)
    B = cv2.calcBackProject([hsvt],[0,1],M,[0,180,0,256],1)
else:
    # Find the histograms using calcHist.
    M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    R = M/I

    # Extract H and S channels from the target image
    h,s,v = cv2.split(hsvt)

    B = R[h.ravel(),s.ravel()]
    B = np.minimum(B,1)
    B = B.reshape(hsvt.shape[:2])

    # Display the histogram
    plt.subplot(2,2,1)
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.imshow(M, interpolation = 'nearest')
    plt.subplot(2,2,2)
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.imshow(I, interpolation = 'nearest')
    plt.subplot(2,2,3)
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.imshow(R, interpolation = 'nearest')
    plt.subplot(2,2,4)
    plt.imshow(B, cmap ='gray')
    plt.show()


# apply a convolution with a circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)

# Use thresholding to segment out the region
ret,thresh = cv2.threshold(B,10,255,0)

# Overlay images using bitwise_and
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)

# Display the output
cv2.imshow('a',res)
cv2.waitKey(0)