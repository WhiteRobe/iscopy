import numpy as np 
import cv2 as cv 
import copy 

def cross_correlation_2d(img,kernel):
    """
       给定任意m*n的权重核，其中m和n必须是奇数
       1.输出与边缘具有相同的维度。
       2.超出图像边界的像素为0.
       3.可能会给出RGB图像，或者灰度图像

    input：
        img：是一个numpy.dnarray,一个可能RGB图像(m * n * 3), 或者一个灰度图像(m * n)
        kernel: 一个2D的numpy.dnarray,不一定相等，但一定都是奇数
    output: 
        返回一个与输入图像维度相同的图像
    """

    kernel=np.array(kernel)
    rowk,colk=kernel.shape



    if img.ndim==3:
        #填充[0,0,0]
        rowi, coli, channel=img.shape
        img=np.pad(img,((rowk//2,rowk//2),(colk//2,colk//2),(0,0)),'constant')
       
        kernel=np.stack([kernel]*3,axis=2)
        #新建一个数组置为0
        dst=np.zeros((rowi,coli,3))
    else:
        rowi, coli = img.shape
        img=np.pad(img,((rowk//2,rowk//2),(colk//2,colk//2)),'constant')
        # img = np.pad(img, ((rowk//2, rowk//2), (colk//2, colk//2)), 'constant',)
        dst=np.zeros((rowi,coli))

    #相关
    for i in range(rowi):
        for j in range(coli):
            sum_arr=img[i:i+rowk,j:j+colk]*kernel
            dst[i,j]=sum(sum(sum_arr))
    return dst

def convolve_2d(img,kernel):
    """  
    使用cross_correlation_2d()进行卷积
    input:
        img: 是一个numpy.dnarray,一个可能的RGB图像(m * n * 3), 或者一个灰度图像(m * n)
        kernel: 一个2D的numpy.dnarray,， 不一定相等，但一定都是奇数
    output: 
        返回一个与输入图像维度相同的图像
    """
    # 最简单的方法，将kernel沿水平和竖直方向反转，然后求相关
    kernel=np.flip(kernel, (0, 1))

    return cross_correlation_2d(img,kernel)

def gaussian_blur_kernel_2d(sigma,height,width):
    """ 
    返回给定维数的高斯模糊核， 高度和宽度是不一样的，但一定都是奇数
    input:
        sigma: 高斯函数的sigma
        width: 内核宽度
        height: 内核的高度
    output:
        返回一个height*width的高斯核
    """
    kernel=np.zeros((height,width),np.float32)
    x_center=height//2
    y_center=width//2

    def get_gaussian(x,y):
        """ 为高斯核的的每一个元素求值的函数，也可以直接表示或者使用lambda表达式
        """
        x=x-x_center
        y=y-y_center
        return np.exp(-0.5*(x**2+y**2)/sigma**2)

    for i in range(height):
        for j in range(width):
            kernel[i,j]=get_gaussian(i,j)

    coe=1/np.sum(kernel)  #归一化
    return coe*kernel

def low_pass(img,sigma,size):
    """ 
    过滤图像， 使用给定的高斯滤波器进行， 
    input: 
        img: 是一个numpy.dnarray,一个可能的RGB图像(m * n * 3), 或者一个灰度图像(m * n）
        sigma: 高斯函数的sigma
        size: 在低通滤波中使用平方核
    output：
        返回一个与输入图像维度相同的图像f
    """
    guassian_kernel=gaussian_blur_kernel_2d(sigma,size,size)

    return convolve_2d(img,guassian_kernel)

def high_pass(img,sigma,size):
    """ 
    过滤图像， 高通滤波，使用img-low_pass(img)
    input: 
        img: 是一个numpy.dnarray,一个可能的RGB图像(m * n * 3), 或者一个灰度图像(m * n）
        sigma: 高斯函数的sigma
        size: 在低通滤波中使用平方核
    output：
        返回一个与输入图像维度相同的图像f
    """

    low_img=low_pass(img,sigma,size)

    return img-low_img


def create_hybrid_image(img1,img2,sigma1,size1,high_low1,sigma2,size2,high_low2,mixin_ration):
    img1=np.array(img1)
    img2=np.array(img2)

    high_low1=high_low1.lower()
    high_low2=high_low2.lower()

    if img1.dtype==np.uint8:
        img1=img1.astype(np.float32)/255.0 
        img2=img2.astype(np.float32)/255.0 

    if high_low1=='low':
        img1=low_pass(img1,sigma1,size1)
    else:
        img1=high_pass(img1,sigma1,size1)

    if high_low2=='low':
        img2=high_pass(img2,sigma2,size2)
    else:
        img2=high_pass(img2,sigma2,size2)

    #print(img1.dtype)
    img1*=2*(1-mixin_ration)
    img2*=2*mixin_ration
    hybrid_img=(img1+img2)
    return (hybrid_img*255).clip(0,255).astype(np.uint8)

if __name__=='__main__':
    left=cv.imread('left.jpg')
    right=cv.imread('right.jpg')
    img=create_hybrid_image(left,right,7.0,13,'low',4.1,8,'high',0.65)
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('hybrid.jpg',img)