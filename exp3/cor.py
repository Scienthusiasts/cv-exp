# activate E:\environments\media_pipe
import numpy as np
import matplotlib.pyplot as plt
import cv2   
import random

import sys;sys.path.append('../')
import utils



# 提取图像ORB特征并转化为numpy数组
def extraORBfromImg(ORB, img):
    keypoints, desc = ORB.detectAndCompute(img, mask=None) # 关键点检测
    # 特征点信息
    axis = np.array([kp.pt for kp in keypoints]) # 特征点图像坐标
    scale = np.array([kp.octave+1 for kp in keypoints])# 特征点尺度(在哪一层金字塔)
    direct = np.array([kp.angle*np.pi/180 for kp in keypoints]) # 特征点方向(弧度)
    # 拼接
    infos = np.array([scale,direct]).T
    cors_info = np.hstack([axis,infos])
    return cors_info, desc



# ORB特征BRIEF描述子匹配
def ORBMatch(BF, desc1, desc2):
    matches = BF.match(desc1, desc2)
    dist = np.array([mc.distance for mc in matches])
    idx1 = np.array([mc.trainIdx for mc in matches])
    idx0 = np.array([mc.queryIdx for mc in matches])
    idx = np.array([idx0,idx1]).T
    # 匹配点筛选，当描述子之间的距离大于两倍的最小距离时，认为匹配有误
    min_dist = min(dist)
    filte_idx = np.where(dist <= max(2 * min_dist, 30))[0]
    return dist[filte_idx], idx[filte_idx,:]


# 获取图像对匹配点的坐标(一对)
def findMatchCord(match_idx, cors1, cors2):
    left = cors1[match_idx[:,0], :2]
    right = cors2[match_idx[:,1], :2]
    return np.hstack([left, right])




# 只利用4对点计算单应性矩阵:
def Homographyfrom4Pts(pair_points, img1, img2):
    pt1 = pair_points[:,:2].astype(np.float32)
    pt2 = pair_points[:,2:4].astype(np.float32)
    # 可能的问题出在三点共线或者两点重合的情况，导致误差巨大
    M = cv2.getPerspectiveTransform(pt1, pt2)
    # 返回的行列式用于辅助检查 M 是否正确
    return M, np.linalg.det(M)




def RANSAC(match_pts, img1, img2):
    pts_num = match_pts.shape[0]
    det_M = 0
    update_match_pts = []
    max_satisfy_rate = 0
    for i in range(100):
        det_M = 0
        while(det_M <= 0.1):
            # 随机选取4对点
            rand4 = np.random.randint(pts_num, size=4)  
            # rand4 = [11, 10, 8, 26]
            # 基于这4对点计算单应性矩阵
            M, det_M = Homographyfrom4Pts(match_pts[rand4,:], img1, img2)

        # 添加齐次坐标
        homo_pts1 = np.insert(match_pts[:,:2], 2, values=np.ones((1, pts_num)), axis=1).T
        # 重投影齐次坐标
        homo_pts2_hat = (M @ homo_pts1).T
        # 重投影坐标
        pts2_hat = (homo_pts2_hat / homo_pts2_hat[:, 2].reshape(-1,1))[:,:2]
        # 计算误差
        error_matrix = np.sum((match_pts[:,2:4] - pts2_hat)**2, axis=1)
        satisfy_rate = sum(error_matrix < 10) / pts_num
        # 若重投影正确率大于当前最大值, 更新认为是正确的匹配点
        if(satisfy_rate > max_satisfy_rate):
            max_satisfy_rate = satisfy_rate
            update_match_pts = match_pts[error_matrix < 10]
        # 若重投影正确率大于阈值, 直接返回结果
        if(satisfy_rate > 0.75):
            return update_match_pts
    return update_match_pts




# 可视化图像对映射效果
def homography_trans(M, img1, img2):
    # out_img 第一张图像映射到第二张
    x_min, x_max, y_min, y_max, M2 = calc_border(M, img1.shape)
    # 透视变换+平移变换(使得图像在正中央)
    M = M2 @ M
    out_img = cv2.warpPerspective(img1, M, (round(x_max)-round(x_min), round(y_max)-round(y_min)))
    # 调整两张图像位姿一致：
    # x方向
    out_img_blank_x = np.zeros((out_img.shape[0], abs(round(x_min)), 3)).astype(np.uint8)
    img2_blank_x = np.zeros((img2.shape[0], abs(round(x_min)), 3)).astype(np.uint8)
    if(x_min>0):
        print(1)
        out_img = cv2.hconcat((out_img_blank_x, out_img))
    if(x_min<0):
        print(2)
        img2 = cv2.hconcat((img2_blank_x, img2))
    # y方向
    out_img_blank_y = np.zeros((abs(round(y_min)), out_img.shape[1], 3)).astype(np.uint8)
    img2_blank_y = np.zeros((abs(round(y_min)), img2.shape[1], 3)).astype(np.uint8)
    if(y_min>0):
        print(3)
        out_img = cv2.vconcat((out_img, out_img_blank_y))
    if(y_min<0):
        print(4)
        img2 = cv2.vconcat((img2_blank_y, img2))
    # 调整两张图像尺度一致：
    if(img2.shape[0]<out_img.shape[0]):
        blank_y = np.zeros((out_img.shape[0]-img2.shape[0], img2.shape[1], 3)).astype(np.uint8)
        img2 = cv2.vconcat((img2, blank_y)) 
    else:
        blank_y = np.zeros((img2.shape[0]-out_img.shape[0], out_img.shape[1], 3)).astype(np.uint8)
        out_img = cv2.vconcat((out_img, blank_y)) 
    if(img2.shape[1]<out_img.shape[1]):
        blank_x = np.zeros((img2.shape[0], out_img.shape[1]-img2.shape[1], 3)).astype(np.uint8)
        img2 = cv2.hconcat((img2, blank_x)) 
    else:
        blank_x = np.zeros((out_img.shape[0], img2.shape[1]-out_img.shape[1], 3)).astype(np.uint8)
        out_img = cv2.hconcat((out_img, blank_x))        

    # cv2.imwrite('out_img.jpg',out_img)
    # 叠加
    result = addMatches(out_img, img2)

    # 叠加显示映射重合度

    mask = 255*np.ones(result.shape).astype(np.uint8)
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    mask[gray_res==0]=0
    # cv2.imshow('vihy',  mask)

    cv2.imwrite('mask.jpg',mask)
    result[result==0]=255
    cv2.imwrite('result.jpg',result)
    return result, out_img


def calc_homography(match_pts):
    src_pts = match_pts[:,:2]
    dst_pts = match_pts[:,2:4]
    # 第三个参数可以使用cv2.RANSAC, 0表示所有点参与计算
    # 第四个参数表示可容忍的重投影误差,范围(1,10)
    # 返回的参数还包含一个mask,参与计算的点数，这里没什么用
    M, _ = cv2.findHomography(src_pts, dst_pts, 0, 10)
    print(M)
    Global_matrix = np.zeros((2 * match_pts.shape[0], 9))
    for i in range(match_pts.shape[0]):
        Global_matrix[2*i,:2] = Global_matrix[2*i+1,3:5] = match_pts[i,:2]
        Global_matrix[2*i,2] = Global_matrix[2*i+1,5] = 1
        Global_matrix[2*i,6] = -match_pts[i,0]*match_pts[i,2]
        Global_matrix[2*i,7] = -match_pts[i,1]*match_pts[i,2]
        Global_matrix[2*i,8] = -match_pts[i,2]
        Global_matrix[2*i+1,6] = -match_pts[i,0]*match_pts[i,3]
        Global_matrix[2*i+1,7] = -match_pts[i,1]*match_pts[i,3]
        Global_matrix[2*i+1,8] = -match_pts[i,3]
    temp_M = Global_matrix.T @ Global_matrix
    # ATA最小特征值对应的特征向量就是齐次方程的解
    eigen_M = np.linalg.eig(temp_M)[1].T[-1].T.reshape(3,3)
    print(np.linalg.eig(temp_M)[1].T[-1])
    M = eigen_M / eigen_M[2,2]
    print(M)

    return M


# 计算边界
def calc_border(M, shape):
    w, h = shape[1], shape[0]
    pt1 = np.array([[0,0],[w,0],[0,h],[w,h]]).astype(np.float32)
    original_border = np.c_[pt1,[1,1,1,1]]
    #计算透视变换后的图像四个角的坐标
    perspected_border = M @ original_border.T
    perspected_border = perspected_border / perspected_border[2,:]
    x_min = min(perspected_border[0,:])
    x_max = max(perspected_border[0,:])
    y_min = min(perspected_border[1,:])
    y_max = max(perspected_border[1,:])
    pt2 = np.array([[-x_min,-y_min],[w-x_min,-y_min],[-x_min,h-y_min],[w-x_min,h-y_min]]).astype(np.float32)
    # 平移变换(将图像平移至正中央防止负坐标变换后被遮挡的情况)
    M2 = cv2.getPerspectiveTransform(pt1, pt2)
    return x_min, x_max, y_min, y_max, M2

# 图像拼接(无脑)
def addMatches(img1, img2):
    img3 = img1[:,:,:]
    img3[img2==0]=img1[img2==0]
    img3[img2!=0]=img2[img2!=0]
    return img3





if __name__ == '__main__':
    # 创建ORB, BRIEF对象
    orb = cv2.ORB_create(1000)
    match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)# crossCheck双向匹配


    img1 = cv2.imread('datas/13.jpg')
    img1,_,_ = utils.auto_reshape(img1, 700)
    img2 = cv2.imread('datas/12.jpg')
    img2,_,_ = utils.auto_reshape(img2, 700)
    cors1, desc1= extraORBfromImg(orb, img1)
    cors2, desc2= extraORBfromImg(orb, img2)
    
    match_dist, match_idx = ORBMatch(match, desc1, desc2)
    # 得到匹配点对的坐标
    match_pts = findMatchCord(match_idx, cors1, cors2)
    # 可视化匹配点
    utils.drawMatches(img1, img2, cors1, cors2, match_idx)
    # RANSAC迭代去除异常点对
    update_match_pts = RANSAC(match_pts, img1, img2)
    # 最小二乘方法计算单应矩阵
    M = calc_homography(update_match_pts)
    homography_trans(M, img1, img2)
    cv2.calibrateCamera()