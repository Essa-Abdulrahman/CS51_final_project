import numpy as np
#from scipy import ndimage
from scipy.ndimage import convolve1d, convolve

import matplotlib.pyplot as plt
#from scipy.ndimage import convolve1d

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def sobel_filters(img):
    ''' input: a gray scale image
       output: the gray scale edge detection image, and direction matrix theta'''
    Kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ky = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    #Ix = ndimage.filters.convolve(img, Kx)
    #Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)

def diff_filters(img):
    ''' input: a gray scale image
       output: the gray scale edge detection image, and direction matrix theta'''
    Dx = np.array([1,0, -1], np.float32).reshape((3,1))
    Dy = np.array([1, 0, -1], np.float32).reshape((1,3))

    Ix = convolve(img, Dx)
    Iy = convolve(img, Dy)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


Dx = np.array([1,0, -1], np.float32).reshape((3,1))
Dy = np.array([1, 0, -1], np.float32).reshape((1,3))

'''
img = plt.imread('face_01.jpg')
img_gray = rgb2gray(img)
img_Dx = convolve(img_gray, Dx)
img_Dy = convolve(img_gray, Dy)
#img_Dx = ndimage.filters.convolve(img_gray,Dx)
#img_Dy = ndimage.filters.convolve(img_gray,Dy)

# draw images
fig = plt.figure(figsize=(12,12))
fig.add_subplot(1,3,1)
plt.imshow(img_gray, cmap='gray')
plt.title("original")
fig.add_subplot(1,3,2)
plt.imshow(img_Dx, cmap='gray')
plt.title("Dx")
fig.add_subplot(1,3,3)
plt.imshow(img_Dy, cmap='gray')
plt.title("Dy")
plt.tight_layout()
plt.show()
'''

img = plt.imread('fig_5.jpg')
img_gray = rgb2gray(img)
img_sobel = sobel_filters(img_gray)[0]

# draw images
fig = plt.figure(figsize=(8,8))
fig.add_subplot(1,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("original")
fig.add_subplot(1,2,2)
plt.imshow(img_sobel, cmap='gray')
plt.title("sobel")
plt.tight_layout()
plt.show()
'''

signal = np.array([[65.0, 129.0, 193.0], [31.0, 130.0, 226.0], [65.0, 129.0, 192.0]], np.float32)
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
out = convolve(signal, kernel)

print(out)
'''