import sys
import cv2
import numpy as np
import math
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    if img.ndim == 3:
        b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
        n = img.shape[0]
        m = img.shape[1]
        (nk, mk) = kernel.shape
        b[:, :] = img[:, :, 0]
        g[:, :] = img[:, :, 1]
        r[:, :] = img[:, :, 2]
        img_new_r = []
        img_new_b = []
        img_new_g = []
        extend_r = np.zeros((nk + n - 1, m + mk - 1), dtype=np.float)
        extend_b = np.zeros((nk + n - 1, m + mk - 1), dtype=np.float)
        extend_g = np.zeros((nk + n - 1, m + mk - 1), dtype=np.float)
        for i in range(n):
            for j in range(m):
                extend_r[i + nk // 2][j + mk // 2] = r[i][j]
                extend_b[i + nk // 2][j + mk // 2] = b[i][j]
                extend_g[i + nk // 2][j + mk // 2] = g[i][j]
        for i in range(n):
            line_r = []
            line_b = []
            line_g = []
            for j in range(m):
                ar = extend_r[i:i + nk, j:j + mk]
                ab = extend_b[i:i + nk, j:j + mk]
                ag = extend_g[i:i + nk, j:j + mk]
                line_r.append(np.sum(np.multiply(ar, kernel)))
                line_b.append(np.sum(np.multiply(ab, kernel)))
                line_g.append(np.sum(np.multiply(ag, kernel)))
            img_new_r.append(line_r)
            img_new_b.append(line_b)
            img_new_g.append(line_g)
        merged = np.dstack([img_new_b,img_new_g,img_new_r])
        return merged
    elif img.ndim == 2:
        n = img.shape[0]
        m = img.shape[1]
        imgnew = []
        (nk, mk) = kernel.shape
        extend = np.zeros((nk + n - 1, m + mk - 1), dtype=np.float)
        for i in range(n):
            for j in range(m):
                extend[i + (nk // 2)][j + (mk // 2)] = img[i][j]
        for i in range(n):
            line = []
            for j in range(m):
                a = extend[i:i + nk, j: j+mk]
                line.append(np.sum(np.multiply(a, kernel)))
            imgnew.append(line)
        return np.asarray(imgnew)
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    ker_new = []
    for i in kernel:
        ker_new.insert(0, i[::-1])
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    return cross_correlation_2d(img, np.asarray(ker_new))


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kerner = np.zeros((height,width), dtype = np.float)
    s = sigma ** 2
    sum_val = 0
    for i in range(height):
        for j in range(width):
            x, y = i - height//2, j - width//2
            kerner[i][j] = np.exp(-(x ** 2 + y ** 2)/(2 * s))*(1/(2*math.pi*s))
            sum_val += kerner[i][j]
    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    return kerner/sum_val


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    height = width = size
    a = gaussian_blur_kernel_2d(sigma, height, width)
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    return convolve_2d(img, a)


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    tmp = low_pass(img, sigma, size)
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    return img - tmp


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
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
    #
    # cv2.imshow('img1', img1)
    cv2.imwrite('left_bur.jpg', (img1 * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite('right_bur.jpg', (img2 * 255).clip(0, 255).astype(np.uint8))
    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio


    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


