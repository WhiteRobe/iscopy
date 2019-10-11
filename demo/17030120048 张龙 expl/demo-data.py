import cv2
import numpy as np
def cross_correlation_2d(img, kernel):
    img_array = np.array(img)
    r = img_array.shape[0]
    c = img_array.shape[1]
    h = img_array.shape[2]
    #内核的高和宽
    r2 = kernel.shape[0]
    c2 = kernel.shape[1]
    #创建新数组
    new1 = np.zeros((r, (int)(c2 / 2)), np.int)  #创建一个长为r,宽为内核一半，dtype类型为int的用0 填充的二维数组
    new2 = np.zeros(((int)(r2 / 2), c + new1.shape[1] * 2), np.int) #创建一个长为内核高一半,宽为第一个数组的长的二倍，dtype类型为int的用0 填充的二维数组
    conv = np.zeros((r, c, h))
    for i in range(3):
        temp_img_array = np.hstack([new1, np.hstack([img_array[:, :, i], new1])])#在水平方向上平铺
        new_img_array = np.vstack([new2, np.vstack([temp_img_array, new2])])#在竖直方向上堆叠
        for j in range(r):
            for k in range(c):
                conv[j][k][i] = min(max(0, (new_img_array[j:j + r2, k:k + c2] * kernel).sum()), 255)
    return conv

    # TODO-BLOCK-BEGIN

    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    kernel2 = np.rot90(np.fliplr(kernel), 2)  # 将图片进行2次逆时针90度翻转
    return cross_correlation_2d(img, kernel2)  # 调用互相关函数

    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
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
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
def low_pass(img, sigma, size):
    kernel = gaussian_blur_kernel_2d(sigma, size.height, size.width)
    return convolve_2d(img, kernel)
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):

    Image = np.array(img)
    return (img - low_pass(Image, sigma, size))
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,high_low2, mixin_ratio):
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
    img2 =img2 * 2 * (1 - mixin_ratio)
    img1 =img1 * 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
class size:
    def __init__(self,height,width):
        self.height = height
        self.width = width
size1 = size(10,10)
size2 = size(14,14)
img1 = cv2.imread('./zfl.jpg')
img2 = cv2.imread('./dsg.jpg')
h,w,_=img1.shape
img2 = cv2.resize(img2,(w,h),interpolation=cv2.INTER_AREA)
ratio = 0.6
cv2.imwrite('left.jpg',img1)
cv2.imwrite('right.jpg',img2)
img_res = create_hybrid_image(img1,img2,4,size1,'low',7,size2,'high',ratio)
cv2.imwrite('hybrid.jpg', img_res)
