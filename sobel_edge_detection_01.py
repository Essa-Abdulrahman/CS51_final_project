import math
import cv2
import numpy as np
from PIL import Image

import scipy.ndimage

def img2rgb(img_file):
    img = Image.open(img_file).convert("RGB")
    pixels = img.load()
    width, height = img.size
    rgb_matrix = [
        [tuple(int(v) for v in pixels[x, y]) for x in range(width)]
        for y in range(height)
    ]
    return rgb_matrix

def rgb2gray(rgb):
    gray = []
    for row in rgb:
        gray_r = []
        for column in row:
            r, g, b = column
            g = 0.299 * r + 0.587 * g + 0.114 * b
            gray_r.append(g)
        gray.append(gray_r)
    return gray

def convolve(signal, kernel):
    #kernel = kernel[::-1][::-1]  # flip kernel
    h = len(signal)
    w = len(signal[0])

    kh = 3
    kw = 3

    pad_h = kh // 2
    pad_w = kw // 2

    # zero-padded image
    padded = [[0 for _ in range(w + 2 * pad_w)] for _ in range(h + 2 * pad_h)]

    for i in range(h):
        for j in range(w):
            padded[i + pad_h][j + pad_w] = signal[i][j]

    output = [[0 for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            total = 0
            for ki in range(kh):
                for kj in range(kw):
                    total += padded[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = total

    return output

def edge_thinning(img, theta):
    row, col = len(img), len(img[0])
    z = [[0] * col for _ in range(row)]
    for i in range(1,row-1):
        for j in range(1,col-1):
            angle = theta[i][j] * 180. / math.pi
            if angle < 0:
                angle += 180
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = img[i+1][j]
                    r = img[i-1][j]
                #angle 45
                elif (22.5 <= angle < 67.5):
                    q = img[i-1][j-1]
                    r = img[i+1][j+1]
                #angle 90
                elif (67.5 <= angle < 112.5):
                    q = img[i][j+1]
                    r = img[i][j-1]
                #angle 135
                elif (112.5 <= angle < 157.5):
                    q = img[i+1][j-1]
                    r = img[i-1][j+1]

                if (img[i][j] >= q) and (img[i][j] >= r):
                    z[i][j] = img[i][j]
                else:
                    z[i][j] = 0

            except IndexError as e:
                pass
    return z

def main():
    mat_x =[[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]

    mat_y =[[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]

    mat_gaus =[ [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]]

    img_location = 'fig_5.jpg'
    img_rgb = img2rgb(img_location)
    img_gray = rgb2gray(img_rgb)
    img_blur = scipy.ndimage.convolve(img_gray, mat_gaus)

    g_x = scipy.ndimage.convolve(img_blur, mat_x)
    g_y = scipy.ndimage.convolve(img_blur, mat_y)
    #g = np.abs(g_x) + np.abs(g_y)
    g = np.hypot(g_x, g_y)
    g = g / g.max() * 255
    theta = np.arctan2(g_y, g_x)
    g = edge_thinning(g, theta)

    img = np.array(g, dtype=np.float32)
    img = np.abs(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    # Save PNG
    cv2.imwrite("fig_0551.png", img)
def main_slow():
    mat_x = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]

    mat_y = [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]

    mat_gaus = [[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]]

    img_location = 'fig_5.jpg'
    img_rgb = img2rgb(img_location)
    img_gray = rgb2gray(img_rgb)
    img_blur = convolve(img_gray, mat_gaus)
    g_x = convolve(img_gray, mat_x)
    g_y = convolve(img_gray, mat_y)

    g = [
        [math.sqrt(g_x[i][j]**2 + g_y[i][j]**2) for j in range(len(g_x[0]))]
        for i in range(len(g_x))
    ]

    g_max = max([max(g[j]) for j in range(len(g))])

    g = [
        [g[i][j]/g_max * 255 for j in range(len(g[0]))]
        for i in range(len(g))
    ]
    theta = [
        [math.atan2(g_y[i][j],g_x[i][j]) for j in range(len(g[0]))]
        for i in range(len(g))
    ]

    g = edge_thinning(g, theta)

    img = np.array(g, dtype=np.float32)
    img = np.abs(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

    # Save PNG
    cv2.imwrite("fig_0551_s.png", img)

if __name__ == "__main__":
    main_slow()