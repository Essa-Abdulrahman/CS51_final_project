import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gaussian_kernel(kernel_size, sigma=1):
    size = int(kernel_size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ky = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)  # x is vertical, y is horizontal - angle measured off x axis.

    return (G, theta)

def edge_thinning(img, D):
    '''input: Intensity Matrix img, and direction matrix D'''
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi   # convert direction matrix from radians to degrees
    angle[angle < 0] += 180 # map all angles from 0 to 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def dbl_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def main_1():
    # load and covert image to gray scale
    img_col = plt.imread("face_02.jpg")
    img = rgb2gray(img_col)
    # apply Gaussian Blur
    img_gb = convolve(img, gaussian_kernel(5))
    # plt.imshow(img_gb, cmap="gray")

    # apply Sobel Filters
    img_sobel, theta = sobel_filters(img_gb)

    # display image
    #plt.imshow(img_sobel, cmap="gray")
    # apply edge thinning
    img_edge = edge_thinning(img_sobel, theta)

    # diplay image
    #plt.imshow(img_edge, cmap="gray")

    # apply Double Threshold
    img_dt, w, s = dbl_threshold(img_edge) #img_nms

    plt.imshow(img_dt, cmap="gray")

    plt.show()

def main():
    # load and covert image to gray scale
    img_col = plt.imread("fig_39.jpg")
    img = rgb2gray(img_col)

    # apply gaussian blur
    img_gb = convolve(img, gaussian_kernel(5))

    # apply Sobel filters
    img_sobel, theta = sobel_filters(img_gb)

    # apply Edge thinning
    img_nms = edge_thinning(img_sobel, theta)

    # apply Double Threshold
    img_dt, weak, strong = dbl_threshold(img_nms)

    # edge tracking by hysteresis
    img_h = hysteresis(img_dt, weak)

    # display images
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(img_sobel, cmap="gray")
    axes[0, 0].set_title("Sobel Filters")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_nms, cmap="gray")
    axes[0, 1].set_title("Edge Thinning")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(img, cmap="gray")
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_h, cmap="gray")
    axes[1, 1].set_title("Edge Detection")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()