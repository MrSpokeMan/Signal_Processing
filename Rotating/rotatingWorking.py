import cv2
from matplotlib import pyplot as plt
import time

fig, ax = plt.subplots(1, 2)


def rotation(angle, flags=cv2.INTER_CUBIC):
    img = cv2.imread('skull.png')
    # convert of color because OpenCV works in BGR and Matplotlib in RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    center = (int(height / 2), int(width / 2))

    rotationMatrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1.0)  # create rotation matrix
    '''
    rotationMatrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    '''

    start = time.time()
    rotatedImage = cv2.warpAffine(src=img, M=rotationMatrix, dsize=(
        height, width), flags=flags)  # making transformation
    end = time.time()
    print(end - start)
    plt.imsave('skull_rotated.png', rotatedImage)

    rotationMatrix2 = cv2.getRotationMatrix2D(
        center=center, angle=-angle, scale=1.0)
    backToNormal = cv2.warpAffine(
        src=rotatedImage, M=rotationMatrix2, dsize=(height, width), flags=flags)
    plt.imsave('backToNormal_rotated.png', backToNormal)

    ax[0] = plt.imshow(rotatedImage)


def resize(scale):
    img = cv2.imread('skull.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    heightScaled = int(height * int(scale))
    widthScaled = int(width * int(scale))

    resizedImage = cv2.resize(src=img, dsize=(
        heightScaled, widthScaled), interpolation=cv2.INTER_LINEAR)
    plt.imsave('skull_resized.png', resizedImage)

    backToNormal = cv2.resize(src=resizedImage, dsize=(
        height, width), interpolation=cv2.INTER_LINEAR)
    plt.imsave('back_normal_resized.png', backToNormal)
    ax[1] = plt.imshow(resizedImage)


angle = int(input("What's the angle of the rotation? "))
rotation(angle)

scale = input("What's the scale: ")
resize(scale)

plt.show()
