import cv2
import numpy as np
import math

img = cv2.imread('./img/leo_noise.jpg')


def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return print(round(10 * np.log10(PIXEL_MAX**2 / mse), 5))


def conv_filter(img, sizeOfKernel):
    kernel = np.ones(sizeOfKernel)
    height, width, dim = img.shape
    kernel_size = np.size(kernel[0])
    suma = np.sum(kernel)
    padded = cv2.copyMakeBorder(img, kernel_size//2, kernel_size //
                                2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)
    result = np.zeros((height, width, dim), dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            area = padded[x:x+kernel_size, y:y+kernel_size, :]
            for i in range(3):
                result[x, y, i] = np.sum(area[:, :, i] * kernel) / suma

    PSNR(img, result)
    cv2.imwrite('./img/conv/result.jpg', result)
    diff = img - result
    cv2.imwrite('./img/conv/diff.jpg', diff)


def median_filter(img, sizeOfKernel):
    height, width, dim = img.shape
    kernel = np.zeros(sizeOfKernel)
    kernel_size = np.size(kernel[0])
    padded = cv2.copyMakeBorder(img, kernel_size//2, kernel_size //
                                2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)

    result = np.zeros((height, width, dim), dtype=np.uint8)

    for x in range(height):
        for y in range(width):
            area = padded[x:x+kernel_size, y:y+kernel_size, :]
            result[x, y, 0] = np.median(area[:, :, 0])
            result[x, y, 1] = np.median(area[:, :, 1])
            result[x, y, 2] = np.median(area[:, :, 2])

    PSNR(img, result)
    cv2.imwrite('./img/median/result.jpg', result)
    diff = img - result
    cv2.imwrite('./img/median/diff.jpg', diff)


# (sigmaColor) - siłę filtrowania dla pikseli w zasięgu
# (sigmaSpace) - siłę filtrowania dla pikseli położonych w odległości
# Właśnie ze względu na powyższe parametry filtr bilateralny jest lepszy od filtra medianowego, ale za to bardziej złożony


def bilateral_filter(img, kernel, sigmaColor, sigmaSpace):
    result = cv2.bilateralFilter(img, kernel, sigmaColor, sigmaSpace)
    PSNR(img, result)
    cv2.imwrite('./img/bilateral/result.jpg', result)
    diff = img - result
    cv2.imwrite('./img/bilateral/diff.jpg', diff)


conv_filter(img, (5, 5))
median_filter(img, (7, 7))
bilateral_filter(img, 25, 75, 75)
