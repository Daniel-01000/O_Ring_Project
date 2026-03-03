import cv2 as cv
import numpy as np
import time


def compute_histogram(img):
    hist = np.zeros(256)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            value = img[x, y]
            hist[value] += 1

    return hist


def dilation(img):
    rows, cols = img.shape
    output = img.copy()

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if img[x, y] == 255:
                output[x-1:x+2, y-1:y+2] = 255

    return output


def erosion(img):
    rows, cols = img.shape
    output = img.copy()

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if np.any(img[x-1:x+2, y-1:y+2] == 0):
                output[x, y] = 0

    return output


# load image
img = cv.imread('images/Orings/Oring1.jpg', 0)

if img is None:
    print("Image not found")
    exit()

copy = img.copy()

# histogram
hist = compute_histogram(copy)

dark_half = hist[0:128]
bright_half = hist[128:256]

first_peak = np.argmax(dark_half)
second_peak = np.argmax(bright_half) + 128

auto_thresh = int((first_peak + second_peak) / 2)

print("Threshold:", auto_thresh)

# thresholding
before = time.time()

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] > auto_thresh:
            img[x, y] = 255
        else:
            img[x, y] = 0

# invert so ring is white
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] == 0:
            img[x, y] = 255
        else:
            img[x, y] = 0

# closing = dilation then erosion
img = dilation(img)
img = erosion(img)

after = time.time()

print("Processing time:", after - before)

cv.imshow("Result", img)
cv.waitKey(0)
cv.destroyAllWindows()