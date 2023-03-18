import numpy as np
import cv2

# Uploading image for bayer and xtrans transformations
img = cv2.imread("./img/strus.bmp")
img_x = cv2.imread("./img/strus.bmp")

# Get necessary info of an image
# dim = img.shape  # tuple (height, width, num of channels)
print(img.shape)
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]


# create bayer matrix to get colors ([0] red, [1] green, [2] blue)
# bayer = np.array([[[0, 1], [0, 0]],
#                   [[1, 0], [0, 1]],
#                   [[0, 0], [1, 0]]]) # 3 dimension

# mosaicking like bayer matrix
for i in range(height):
    for j in range(width):
        if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):  # green
            img[i][j] = [0, img[i][j][1], 0]
        elif i % 2 == 0 and j % 2 == 1:  # blue
            img[i][j] = [0, 0, img[i][j][2]]
        elif i % 2 == 1 and j % 2 == 0:  # red
            img[i][j] = [img[i][j][0], 0, 0]
# mosaicking like x trans matrix
for i in range(height):
    for j in range(width):
        if ((i % 6 == 0 or i % 6 == 3) and (j % 6 == 0 or j % 6 == 3)) or ((i % 6 == 1 or i % 6 == 2 or i % 6 == 4 or i % 6 == 5) and (j % 6 == 1 or j % 6 == 2 or j % 6 == 4 or j % 6 == 5)):  # green
            img_x[i][j] = [0, img_x[i][j][1], 0]
        elif (i % 6 == 0 and (j % 6 == 2 or j % 6 == 4)) or ((i % 6 == 1 or i % 6 == 5) and j % 6 == 0) or ((i % 6 == 2 or i % 6 == 4) and j % 6 == 3) or (i % 6 == 3 and (j % 6 == 1 or j % 6 == 5)):  # blue
            img_x[i][j] = [0, 0, img_x[i][j][2]]
        elif ((i % 6 == 1 or i % 6 == 5) and j % 6 == 3) or ((i % 6 == 2 or i % 6 == 4) and j % 6 == 0) or (i % 6 == 3 and (j % 6 == 2 or j % 6 == 4)) or (i % 6 == 0 and (j % 6 == 1 or j % 6 == 5)):  # red
            img_x[i][j] = [img_x[i][j][0], 0, 0]

# sum = np.concatenate((img, img_x), axis=1)
# # Opening image, first argument as name of the window
# cv2.imshow("Images", sum)
# Saving modified image
cv2.imwrite("./img/raw.bmp", img)
cv2.imwrite("./img/raw_x.bmp", img_x)

# # Preventing closing a window after time
# cv2.waitKey(0)
# # cleaning memory after closing window
# cv2.destroyAllWindows()

# interpolation for bayer
deBayerMask = np.ones((2, 2))
deBayerFilter = [deBayerMask * w for w in [1, 1/2, 1]]

R, G, B = [cv2.filter2D(img[..., i], -1, deBayerFilter[i]) for i in range(3)]
rgb = np.dstack((R, G, B))

cv2.imwrite("./img/post_bayer.bmp", rgb)

# interpolation for xtrans
deTransMask = np.array([[0., 0., 0., 0., 0., 0.],
                        [0., 0.25, 0.5, 0.5, 0.25, 0.],
                        [0., 0.5, 1., 1., 0.5, 0.],
                        [0., 0.5, 1., 1., 0.5, 0.],
                        [0., 0.25, 0.5, 0.5, 0.25, 0.],
                        [0., 0., 0., 0., 0., 0.]])
deTransFilter = np.array([deTransMask * i for i in [1/2, 1/5, 1/2]])

Rx, Gx, Bx = [cv2.filter2D(img_x[..., n], -1, deTransFilter[n])
              for n in range(3)]
rgb_x = np.dstack((Rx, Gx, Bx))

cv2.imwrite("./img/post_xtrans.bmp", rgb_x)

# show difference
img = img-rgb

cv2.imwrite("./img/diff_bayer.bmp", img)

img_x = img_x - rgb_x

cv2.imwrite("./img/diff_xtrans.bmp", img_x)
