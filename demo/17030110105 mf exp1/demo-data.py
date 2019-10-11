
import cv2
import numpy as np


def cross_correlation_2d(img, kernel):  # 互相关
    img_array = np.array(img)  # 把图像转换为数字
    r = img_array.shape[0]
    c = img_array.shape[1]  # 图像的列
    h = img_array.shape[2]  # 图像的高度
    r2 = kernel.shape[0]  # 核的行
    c2 = kernel.shape[1]  # 核的列
    new1 = np.zeros((r, (int)(c2 / 2)), np.int)  # 获得一个新的空白矩阵
    new2 = np.zeros(((int)(r2 / 2), c + new1.shape[1] * 2), np.int)
    conv = np.zeros((r, c, h))
    for i in range(3):  # 对矩阵进行一个互相关运算
        temp_img_array = np.hstack([new1, np.hstack([img_array[:, :, i], new1])])  # 对函数增加一个维度
        new_img_array = np.vstack([new2, np.vstack([temp_img_array, new2])])
        for j in range(r):
            for k in range(c):
                conv[j][k][i] = min(max(0, (new_img_array[j:j + r2, k:k + c2] * kernel).sum()), 255)
    return conv


def convolve_2d(img, kernel):  # 卷积
    kernel2 = np.rot90(np.fliplr(kernel), 2)  # 将图片进行2次逆时针90度翻转
    return cross_correlation_2d(img, kernel2)  # 调用互相关函数


def gaussian_blur_kernel_2d(sigma, height, width):  # 产生一个高斯核
    gaussian_kernel = np.zeros((height, width), dtype='double')
    center_row = height / 2
    center_column = width / 2
    s = 2 * (sigma ** 2)
    for i in range(height):
        for j in range(width):
            x = i - center_row
            y = j - center_column
            gaussian_kernel[i][j] = (1.0 / (np.pi * s)) * np.exp(-float(x ** 2 + y ** 2) / s)
    return gaussian_kernel/gaussian_kernel.sum()  # 返回高斯核


# def print_gaussian(gaussian,height,width):
#     for i in range(height):  #s输出核函数
#         for j in range(width):
#             print(gaussian[i][j]end)
#         print(' \n')
#     print('\n')  # 在经过一次低通核函数输出以后输出高通的核函数
#     print('\n')  # 在经过一次低通核函数输出以后输出高通的核函数
#     print('\n')  # 在经过一次低通核函数输出以后输出高通的核函数
#     print('\n')  # 在经过一次低通核函数输出以后输出高通的核函数

def low_pass(img, sigma, size):
    res = gaussian_blur_kernel_2d(sigma,size,size) # res为一个高斯核
    #    print_gaussian(res,height,width)#把核函数输出来看看
    return convolve_2d(img, res)  # 进行卷积


def high_pass(img, sigma,size):
    Image = np.array(img)
    return (img - low_pass(Image, sigma, size))  # 做一个减法得到高通图像


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


img1 = cv2.imread("C:\\Users\\hh\\Desktop\\leftimg.jpg")
img2 = cv2.imread("C:\\Users\\hh\\Desktop\\rightimg.jpg")
ratio = 0.6
img2_res = low_pass(img2, 5, 13)  # 得到低通图像
img2_res=img2_res.astype(np.float32)/255.0
img1_res = high_pass(img1, 15, 25)  # 得到高通图像
img1_res=(img1_res.astype(np.float32)/255.0)*2.4
#img3 = create_hybrid_image(img1,img2,15,25,25,5,13,13,0.6)
cv2.imwrite('left.jpg', img1_res)
cv2.imwrite('right.jpg', img2_res)
#img_res = cv2.addWeighted(img1_res, ratio, img2_res, ratio, 0)  # 图像混合加权函数
#cv2.imwrite('hybrid.jpg', img_res)
cv2.namedWindow("StrikeFreedom")
cv2.namedWindow("Destiny")
cv2.imshow("Destiny",img2_res)
cv2.imshow("StrikeFreedom",img1_res)
#cv2.nanedwindow("mix")
#cv2.imshow("mix",img3)
img3=create_hybrid_image(img1,img2,15,25,"high",5,13,"low",0.6)

cv2.imwrite('img3.jpg',img3)
cv2.waitKey(0) #等待键盘触发事件，释放窗口

