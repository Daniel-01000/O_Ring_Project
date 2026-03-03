import cv2 as cv
import numpy as np
import time


def compute_histogram(img):
    hist = np.zeros(256)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            hist[img[x, y]] += 1

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


def measure_thickness(img):
    rows, cols = img.shape
    thickness_values = []

    for x in range(rows):

        first_white = None
        inner_edge = None

        for y in range(cols):
            if img[x, y] == 255:
                first_white = y
                break

        if first_white is None:
            continue

        for y in range(first_white + 1, cols):
            if img[x, y] == 0:
                inner_edge = y
                break

        if inner_edge is not None:
            thickness_values.append(inner_edge - first_white)

    return thickness_values



img = cv.imread('images/Orings/Oring1.jpg', 0)

if img is None:
    print("Image not found")
    exit()

start = time.time()

# Histogram threshold
hist = compute_histogram(img.copy())

dark_half = hist[0:128]
bright_half = hist[128:256]

first_peak = np.argmax(dark_half)
second_peak = np.argmax(bright_half) + 128

threshold = int((first_peak + second_peak) / 2)

# Manual thresholding
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if img[x, y] > threshold:
            img[x, y] = 255
        else:
            img[x, y] = 0

# Invert so ring is white
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        img[x, y] = 255 - img[x, y]

# Binary closing
img = dilation(img)
img = erosion(img)

# Connected components
labels, sizes = connected_components(img)

largest_label = max(sizes, key=sizes.get)

rows, cols = img.shape
output = np.zeros((rows, cols))

for x in range(rows):
    for y in range(cols):
        if labels[x, y] == largest_label:
            output[x, y] = 255

# Thickness analysis
thickness = measure_thickness(output)

min_t = min(thickness)
max_t = max(thickness)

if max_t - min_t > 40:
    result = "FAIL"
else:
    result = "PASS"

end = time.time()

print("Threshold:", threshold)
print("Ring area:", sizes[largest_label])
print("Min thickness:", min_t)
print("Max thickness:", max_t)
print("Final Result:", result)
print("Processing time:", end - start)

cv.putText(output.astype(np.uint8),
           f"Result: {result}",
           (20, 30),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255),
           2)

cv.putText(output.astype(np.uint8),
           f"Time: {round(end - start, 4)}s",
           (20, 60),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255),
           2)

cv.imshow("Final Result", output.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()