import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Load the image
img = cv2.imread('opencv_logo.png')

########################################
#             Using Scikit             #
########################################
# Add salt-and-pepper noise to the image.
noise_img_gaussian = random_noise(img, mode='gaussian')
noise_img_localvar = random_noise(img, mode='localvar')
noise_img_poisson = random_noise(img, mode='poisson')
noise_img_salt = random_noise(img, mode='salt',amount=0.3)
noise_img_pepper = random_noise(img, mode='pepper',amount=0.3)
noise_img_s_p = random_noise(img, mode='s&p',amount=0.3)
noise_img_speckle = random_noise(img, mode='speckle')

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
noise_img_gaussian = np.array(255*noise_img_gaussian, dtype = 'uint8')
noise_img_localvar = np.array(255*noise_img_localvar, dtype = 'uint8')
noise_img_poisson = np.array(255*noise_img_poisson, dtype = 'uint8')
noise_img_salt = np.array(255*noise_img_salt, dtype = 'uint8')
noise_img_pepper = np.array(255*noise_img_pepper, dtype = 'uint8')
noise_img_s_p = np.array(255*noise_img_s_p, dtype = 'uint8')
noise_img_speckle = np.array(255*noise_img_speckle, dtype = 'uint8')

# Display the noise image
plt.subplot(2,4,1)
plt.title("gaussian")
plt.imshow(noise_img_gaussian)

plt.subplot(2,4,2)
plt.title("localvar")
plt.imshow(noise_img_localvar)

plt.subplot(2,4,3)
plt.title("poisson")
plt.imshow(noise_img_poisson)

plt.subplot(2,4,4)
plt.title("salt")
plt.imshow(noise_img_salt)

plt.subplot(2,4,5)
plt.title("pepper")
plt.imshow(noise_img_pepper)

plt.subplot(2,4,6)
plt.title("s&p")
plt.imshow(noise_img_s_p)

plt.subplot(2,4,7)
plt.title("speckle")
plt.imshow(noise_img_speckle)
plt.show()

########################################
#             Using Numpy              #
########################################
# Generate Gaussian noise
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
# Add the Gaussian noise to the image
img_gauss = cv2.add(img,gauss)

# Generate Speckle noise
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
img_speckle = img + img * gauss
 
plt.subplot(1,2,1)
plt.title("gaussian")
plt.imshow(img_gauss)

plt.subplot(1,2,2)
plt.title("speckle")
plt.imshow(img_speckle)
plt.show()