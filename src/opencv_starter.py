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


def connected_components(img):
    rows, cols = img.shape
    labels = np.zeros((rows, cols))
    current_label = 1
    sizes = {}

    for x in range(rows):
        for y in range(cols):

            if img[x, y] == 255 and labels[x, y] == 0:

                stack = [(x, y)]
                sizes[current_label] = 0

                while stack:
                    i, j = stack.pop()

                    if (0 <= i < rows and 0 <= j < cols and
                        img[i, j] == 255 and labels[i, j] == 0):

                        labels[i, j] = current_label
                        sizes[current_label] += 1

                        stack.append((i+1, j))
                        stack.append((i-1, j))
                        stack.append((i, j+1))
                        stack.append((i, j-1))

                current_label += 1

    return labels, sizes




img = cv.imread('images/Orings/Oring1.jpg', 0)

if img is None:
    print("Image not found")
    exit()

copy = img.copy()

# Histogram threshold
hist = compute_histogram(copy)

dark_half = hist[0:128]
bright_half = hist[128:256]

first_peak = np.argmax(dark_half)
second_peak = np.argmax(bright_half) + 128

auto_thresh = int((first_peak + second_peak) / 2)

print("Threshold:", auto_thresh)

before = time.time()

# Manual threshold
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] > auto_thresh:
            img[x, y] = 255
        else:
            img[x, y] = 0

# Invert so ring is white
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] == 0:
            img[x, y] = 255
        else:
            img[x, y] = 0

# Binary closing
img = dilation(img)
img = erosion(img)

# Connected components
labels, sizes = connected_components(img)

print("Regions found:", len(sizes))
print("Region sizes:", sizes)

# Keep largest region only
largest_label = max(sizes, key=sizes.get)

rows, cols = img.shape
output = np.zeros((rows, cols))

for x in range(rows):
    for y in range(cols):
        if labels[x, y] == largest_label:
            output[x, y] = 255

after = time.time()

print("Processing time:", after - before)

cv.imshow("Final Result", output.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()