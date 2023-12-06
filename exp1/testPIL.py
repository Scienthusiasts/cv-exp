import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

import sys;sys.path.append('../')
import utils

img = cv2.imread('luda04.jpg')[:,:, [0]]
# 图像reshape
img, h, w = utils.auto_reshape(img, 480)
# # 计算直方图
# img_hist = utils.histogram(img)
# # 绘制直方图
# utils.plot_hist(img_hist)
# 直方图均衡化
# eqhist_img = utils.histEqualize(img)
# 计算直方图
img_eqhist = utils.histogram(img)
# 绘制直方图
utils.plot_hist(img, img_eqhist)

# 结果对比
# imgseries = {'raw img':img, 'eqhist img':eqhist_img}
# utils.view_contrast(imgseries)