import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import math
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    # TODO-BLOCK-BEGIN

    m, n = kernel.shape[0:2]
    height, width = img.shape[0:2]
    if img.ndim == 3:
        img = np.pad(img, ((m // 2, m // 2), (n // 2, n // 2), (0, 0)), 'constant')
        new_height, new_width = img.shape[0:2]
        B = np.zeros([height, width], dtype=img.dtype)
        G = np.zeros([height, width], dtype=img.dtype)
        R = np.zeros([height, width], dtype=img.dtype)
        for i in range(height):
            for j in range(width):
                B[i, j] = np.sum(img[i:i + m, j:j + n, 0] * kernel)
                G[i, j] = np.sum(img[i:i + m, j:j + n, 1] * kernel)
                R[i, j] = np.sum(img[i:i + m, j:j + n, 2] * kernel)
        B = B.clip(0, 255)
        G = G.clip(0, 255)
        R = R.clip(0, 255)
        new_img = np.dstack([B, G, R])
        return new_img
    elif img.ndim == 2:
        img = np.pad(img, ((m // 2, m // 2), (n // 2, n // 2)), 'constant')
        height, width = img.shape[0:2]
        new_h = height - m + 1
        new_w = width - n + 1
        new_img = np.zeros([new_h, new_w], dtype=img.dtype)
        for i in range(new_h):
            for j in range(new_w):
                new_img[i, j] = np.sum(img[i:i + m, j:j + n] * kernel)
        new_img = new_img.clip(0, 255)
        return new_img

    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    # TODO-BLOCK-BEGIN
    kernel2 = np.rot90(np.fliplr(kernel), 2)  # 将图片进行2次逆时针90度翻转
    return cross_correlation_2d(img, kernel2)  # 调用互相关函数
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    # TODO-BLOCK-BEGIN
    gaussian_kernel = np.zeros((height, width), dtype='double')
    center_row = height / 2
    center_column = width / 2
    s = 2 * (sigma ** 2)
    for i in range(height):
        for j in range(width):
            x = i - center_row
            y = j - center_column
            gaussian_kernel[i][j] = (1.0 / (np.pi * s)) * np.exp(-float(x ** 2 + y ** 2) / s)
    return gaussian_kernel  # 返回高斯核
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END


def low_pass(img, sigma, size):
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    # TODO-BLOCK-BEGIN
    return img - low_pass(img, sigma, size)
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)



