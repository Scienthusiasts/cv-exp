from wsgiref import util
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np   
import os
import cv2
from tqdm import tqdm, trange
import pydot

import sys;sys.path.append('../')
import utils


data_root, result_root = './datasets/', './cor_info/'
img_path, img_name = utils.read_img_files(data_root)
img1 = cv2.imread('./1.png')[:,:,[2,1,0]]
img2 = cv2.imread('./2.png')[:,:,[2,1,0]]
# img3 = cv2.imread('./corner0.png')[:,:,[2,1,0]]
imgs = {'img1':img1,'img2':img2,}
utils.view_contrast(imgs)