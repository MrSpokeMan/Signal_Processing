from matplotlib import image
from matplotlib import pyplot as plt
import math
import numpy as np
from PIL import Image
import cv2


def resizing(original_img, new_height, new_width):
    old_height, old_width, c = original_img.shape  # get original size of image
    # Creating empty array of the desire shape
    resized = np.zeros((new_height, new_width, c))
    # Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_width) / (new_width) if new_height != 0 else 0
    h_scale_factor = (old_height) / (new_height) if new_width != 0 else 0
    for i in range(new_height):
        for j in range(new_width):
            # map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            # calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(old_height - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_width - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i, j, :] = q
    return resized.astype(np.uint8)


def rotate(image, degree):
    rads = math.radians(degree)
    center_x, center_y = (image.shape[1]//2, image.shape[0]//2)
    new_height = round(abs(
        image.shape[0] * math.sin(rads))) + round(abs(image.shape[1] * math.cos(rads)))
    new_width = round(abs(image.shape[1] * math.cos(rads))) + \
        round(abs(image.shape[0] * math.sin(rads)))
    rotated_image = np.uint8(np.zeros((new_height, new_width, image.shape[2])))
    rotated_center_x, rotated_center_y = (new_width//2, new_height//2)
    for i in range(rotated_image.shape[0]):
        for j in range(rotated_image.shape[1]):
            x = (i - rotated_center_x) * math.cos(rads) + \
                (j - rotated_center_y) * math.sin(rads)
            y = -(i - rotated_center_x) * math.sin(rads) + \
                (j - rotated_center_y) * math.cos(rads)
            x = round(x) + center_x
            y = round(y) + center_y
            if (x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]):
                rotated_image[i, j, :] = image[x, y, :]
    return rotated_image


img = cv2.imread('fortnite.png')
img1 = Image.open('fortnite.png')
img2 = image.imread('fortnite.png')
# downScaled = downScaling(img1)
# upScaled = upScaling(downScaled)
# downScaled.show()
# upScaled.show()
# resized = cv2.resize(img, (240,240), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite('cubicDown.png', resized)
# resized = cv2.resize(resized, (480,480), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite('cubicUp.png', resized)
#d = bl_resize(img, 240, 240)
#cv2.imwrite("linearDown.png", d)
# cv2.imshow('frame', d)
#d = bl_resize(d, 480, 480)
#cv2.imwrite("linearUp.png", d)
# cv2.imshow('frame1', d)
# cv2.waitKey(0)

a = rotate(img, 20)
b = rotate(img, 130)
c = resizing(img, 1200, 1200)
cv2.imwrite('resized.png', c)
cv2.imwrite('20d.png', a)
cv2.imwrite('130d.png', b)
cv2.waitKey(0)
#value = PSNR(img, resized)
#print(f"PSNR value is {value} dB - scaled up")
