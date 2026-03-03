import cv2 as cv
import numpy as np
import time


# count how many times each gray value appears
def compute_histogram(img):
    hist = np.zeros(256)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            value = img[x, y]
            hist[value] += 1

    return hist


# load image in grayscale
img = cv.imread('images/Orings/Oring1.jpg', 0)

if img is None:
    print("Image not found")
    exit()

copy = img.copy()

# build histogram
hist = compute_histogram(copy)

# split into dark and bright parts
dark_half = hist[0:128]
bright_half = hist[128:256]

# find main peak in each half
first_peak = np.argmax(dark_half)
second_peak = np.argmax(bright_half) + 128

# choose threshold between peaks
auto_thresh = int((first_peak + second_peak) / 2)

print("Dark peak:", first_peak)
print("Bright peak:", second_peak)
print("Threshold:", auto_thresh)

# manual thresholding
before = time.time()

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] > auto_thresh:
            img[x, y] = 255
        else:
            img[x, y] = 0

after = time.time()

print("Processing time:", after - before)

cv.imshow("Thresholded Image", img)
cv.waitKey(0)
cv.destroyAllWindows()