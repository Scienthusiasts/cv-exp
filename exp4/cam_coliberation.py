# E:\environments\media_pipe

import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

import sys;sys.path.append('../')
import utils
# 代码中设置环境变量
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "D:\YHT\颜浩天\stereo\stereo_env\Lib\site-packages\PyQt5\Qt5\plugins"


# 定位角点
def find_chessboard_cor(img, w, h):
    # 转为灰度图
    # print(img.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OpenCV内置函数提取棋盘格角点, (11,8)为棋盘格尺寸-1(12x9)
    is_success, corner = cv2.findChessboardCorners(gray_img, (w, h), None)
    # 计算亚像素时停止迭代的标准
    # 后者表示迭代次数达到了最大次数时停止，前者表示角点位置变化的最小值已经达到最小时停止迭代
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # 亚像素角点检测，提高角点定位精度, (7, 7)为考虑角点周围区域的大小
    corner = cv2.cornerSubPix(gray_img, corner, (9, 9), (-1, -1), criteria) 
    return is_success, corner


# 可视化角点
def draw_chessboard_cor(img, cor, is_success):
    cv2.drawChessboardCorners(img, (11,8), cor, is_success)
    cv2.imshow('corcor', img)
    cv2.waitKey(50)



# 旋转向量转旋转矩阵
def Rodriguez(rvecs):
    # 旋转向量模长
    θ = (rvecs[0] * rvecs[0] + rvecs[1] * rvecs[1] + rvecs[2] * rvecs[2])**(1/2)
    # 旋转向量的单位向量
    r = rvecs / θ
    # 旋转向量单位向量的反对称矩阵
    anti_r = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])
    # 旋转向量转旋转矩阵(Rodriguez公式)     # np.outer(r, r) = r @ r.T 向量外积
    M = np.eye(3) * np.cos(θ) + (1 - np.cos(θ)) * np.outer(r, r) + np.sin(θ) * anti_r
    return M


# 可视化标定过程中的相机位姿
def show_cam_pose(rvecs, tvecs):
    # 相机坐标系下基向量
    vec = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])  
    # 相机位姿模型
    cam = (2/3)*np.array([
        [ 1, 1, 2],
        [-1, 1, 2],
        [-1,-1, 2],
        [ 1,-1, 2],
        [ 1, 1, 2],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制每一个角度拍摄的相机
    for i in range(rvecs.shape[0]): 
        # 旋转向量转旋转矩阵
        M = Rodriguez(rvecs[i,:])
        # 相机原点 w = R^(-1)(0 - t)
        x0,y0,z0 = M.T @ (-tvecs[i,:])
        c = ['r','g','b']
        # 随机颜色
        hex = '0123456789abcdef'
        rand_c = '#'+''.join([hex[random.randint(0,15)] for _ in range(6)])
        # 绘制相机坐标系
        for j in range(3):
            # 相机位姿(相机坐标系转世界坐标系)
            # w = R^(-1)(c - t)
            x1,y1,z1 = M.T @ (vec[j,:] - tvecs[i,:])
            # 相机坐标系
            ax.plot([x0,x1],[y0,y1],[z0,z1],color=c[j])
        C = (M.T @ (cam - tvecs[i,:]).T).T 
        # 绘制相机位姿
        for k in range(4):
            ax.plot([C[k,0],C[k+1,0]],[C[k,1],C[k+1,1]],[C[k,2],C[k+1,2]], color=rand_c)
            ax.plot([x0,C[k+1,0]],[y0,C[k+1,1]],[z0,C[k+1,2]], color=rand_c)
        # 相机编号
        ax.text(x0,y0,z0,i+1)
    # 绘制棋盘格
    for i in range(9):
        ax.plot([0, 11],[-i, -i],[0,0],color="black")
    for i in range(12):
        ax.plot([i, i],[-8, 0],[0,0],color="black")
    # 绘制世界(棋盘格)坐标系
    for i in range(3):
        ax.plot([0, 3*vec[i,0]],[0,  -3*vec[i,1]],[0, 2*vec[i,2]],color=c[i],linewidth=3)
    plt.xlim(-3,14)
    plt.ylim(-13,4)



# 可视化标定过程中的棋盘位姿
def show_chessboard_pose(rvecs, tvecs):
    # 棋盘坐标系下基向量
    vec = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])  
    # 相机位姿模型
    C = np.array([
        [ 1, 1, 3],
        [-1, 1, 3],
        [-1,-1, 3],
        [ 1,-1, 3],
        [ 1, 1, 3],
    ])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['r','g','b'] # 坐标轴颜色
    # 绘制每一个角度拍摄的棋盘格
    for i in range(rvecs.shape[0]): 
        # 旋转向量转旋转矩阵
        M = Rodriguez(rvecs[i,:])
        # 棋盘原点 c = Rw + t
        x0,y0,z0 = tvecs[i,:]
        # 随机颜色
        hex = '0123456789abcdef'
        rand_c = '#'+''.join([hex[random.randint(0,15)] for _ in range(6)])
        # 绘制棋盘位姿
        for k in range(0,9):
            b = np.array([[0, -k, 0],[11, -k, 0]])
            b = (M @ b.T + tvecs[i,:].reshape(-1,1)).T
            ax.plot([b[0,0], b[1,0]],[b[0,1], b[1,1]],[b[0,2], b[1,2]],color=rand_c)
        for k in range(0,12):
            b = np.array([[k, -8, 0],[k, 0, 0]])
            b = (M @ b.T + tvecs[i,:].reshape(-1,1)).T
            ax.plot([b[0,0], b[1,0]],[b[0,1], b[1,1]],[b[0,2], b[1,2]],color=rand_c)
            k += 11
        # 绘制棋盘坐标系
        for j in range(3):
            # (世界坐标系转相机坐标系)
            # c = Rw + t
            x1,y1,z1 = M @ vec[j,:] + tvecs[i,:]
            ax.plot([x0,x1],[y0,y1],[z0,z1],color=c[j])
        # 棋盘编号
        ax.text(x0,y0,z0,i+1)
    # 绘制世界(相机)坐标系
    for i in range(3):
        ax.plot([0, vec[i,0]],[0,  -vec[i,1]],[0, 2*vec[i,2]],color=c[i],linewidth=2)
    # 绘制相机位姿
    for i in range(4):
        ax.plot([C[i,0],C[i+1,0]],[C[i,1],C[i+1,1]],[C[i,2],C[i+1,2]], color="black")
        ax.plot([0,C[i+1,0]],[0,C[i+1,1]],[0,C[i+1,2]], color="black")









# 求解内外参数
def CamCalibrate(ratio, w, h, num, root):
    # 图像缩放比例(如果你的图像进行了缩放，与实际拍摄的分辨率不一致，最终求得的参数需要乘上这个比例进行校正) 
    world, cam = [], []

    # 多张图像进行标定，减小误差:
    for i in range(num):
        img = cv2.imread(root + str(i+1)+'.jpg')
        # img,_,_ = utils.auto_reshape(img, 1920)
        # 定位角点
        is_success, cam_coord = find_chessboard_cor(img, w, h)
        print('第'+str(i+1)+'张角点提取完毕, 角点数 =',cam_coord.shape[0])
        # 可视化角点
        # draw_chessboard_cor(img, cam_coord, is_success)
        # 角点的世界坐标:
        # 注:相机参数的计算只要求角点之间的世界坐标比例一致,因此可以单位化
        world_coord = np.zeros((w * h, 3), np.float32)
        world_coord[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        world_coord[:,1] = -world_coord[:,1]
        # world_coord[:,:2] = np.mgrid[0:w*len:len,0:h*len:len].T.reshape(-1,2)
        # 将世界坐标与像素坐标加入待求解系数矩阵
        world.append(world_coord)
        cam.append(cam_coord)


    # 求解摄像机的内在参数和外在参数
    # ret 非0表示标定成功 mtx 内参数矩阵，dist 畸变系数，rvecs 旋转向量，tvecs 平移向量
    # 注:求解的结果的单位为像素,若想化为度量单位还需乘上每个像素代表的实际尺寸(如:毫米/像素)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world, cam, (img.shape[1], img.shape[0]), None, None)
    rvecs = np.array(rvecs).reshape(-1,3)
    tvecs = np.array(tvecs).reshape(-1,3)
    # 单位:像素(1像素=??mm)
    print("标定结果 ret:", ret)
    print("内参矩阵 mtx:\n", mtx * ratio)    # 内参数矩阵
    print("畸变系数 dist:\n", dist)   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("旋转向量(外参) rvecs:\n", rvecs)  # 旋转向量  # 外参数(欧拉角)
    print("平移向量(外参) tvecs:\n", tvecs)  # 平移向量  # 外参数
    
    np.save('./param/mtx.npy',mtx * ratio)
    np.save('./param/dist.npy',dist)
    np.save('./param/rvecs.npy',rvecs)
    np.save('./param/tvecs.npy',tvecs)
    np.save('./param/world.npy',np.array(world))
    np.save('./param/cam.npy',np.array(cam))
    return ret, mtx, dist, rvecs, tvecs





# 计算重投影误差:
def CalcReprojError(num, world, cam, mtx, dist, rvecs, tvecs):
    errors = []
    for i in range(num):
        # 计算重投影坐标(dist=0不考虑畸变)
        # reproj, _ = cv2.projectPoints(world[i], rvecs[i], tvecs[i], mtx, dist)
        reproj = CalcProjectPoints(world[i], rvecs[i], tvecs[i], mtx, dist)
        # 原始坐标
        original_pts = cam[i].reshape(cam[i].shape[0], 2)
        # 重投影坐标
        reprojec_pts = reproj.reshape(reproj.shape[0], 2)
        # RMSE
        error = original_pts - reprojec_pts
        error = np.sum(error*error)**(1/2) / reproj.shape[0]
        # 等价:
        # error = cv2.norm(cam[i],reproj, cv2.NORM_L2) / reproj.shape[0]
        errors.append(error)
        # 重投影可视化
        img = cv2.imread('./0/' + str(i+1)+'.jpg')
        # img,_,_ = utils.auto_reshape(img, 1920)
        drawReprojCor(img, original_pts, reprojec_pts, i)

    # 误差条形图
    # 可视化每张图像的误差(单位:像素)
    plt.bar(range(num), errors, width=0.8, label='reproject error', color='#87cefa')
    # 误差平均值(单位:像素)
    mean_error = sum(errors) / num
    plt.plot([-1,num], [mean_error,mean_error], color='r', linestyle='--',label='overall RMSE:%.3f'%(mean_error))
    plt.xticks(range(num), range(1,num+1))
    plt.ylabel('RMSE Error in Pixels')
    plt.xlabel('Images')
    plt.legend()







# 计算重投影坐标 
def CalcProjectPoints(world, rvecs, tvecs, mtx, dist):
    # 旋转向量转旋转矩阵
    M = Rodriguez(rvecs)
    # c = Rw + t (世界坐标系转相机坐标系)
    R_t =  (M @ world.T).T + tvecs
    # (相机坐标系到图像坐标系)
    plain_pts = (mtx @ R_t.T) #(3, 88) [X, Y, Z]
    plain_pts = (plain_pts / plain_pts[2,:]).T[:,:2] # (88, 2) [X/Z, Y/Z, 1(舍去)]
    # 去畸变
    c_xy = np.array([mtx[0,2], mtx[1,2]])
    f_xy = np.array([mtx[0,0], mtx[1,1]])
    
    k1, k2, p1, p2, k3 = dist[0]
    x_y = (plain_pts - c_xy) / f_xy
    r = np.sum(x_y * x_y, 1)

    x_distorted = x_y[:,0] * (1 + k1*r + k2*r*r + k3*r*r*r) + 2*p1*x_y[:,0]*x_y[:,1] + p2*(r + 2*x_y[:,0]*x_y[:,0])
    y_distorted = x_y[:,1] * (1 + k1*r + k2*r*r + k3*r*r*r) + 2*p2*x_y[:,0]*x_y[:,1] + p1*(r + 2*x_y[:,1]*x_y[:,1])
    u_distorted = f_xy[0]*x_distorted + c_xy[0]
    v_distorted = f_xy[1]*y_distorted + c_xy[1]
    plain_pts = np.array([u_distorted,v_distorted]).T
    return plain_pts









# 重投影可视化
def drawReprojCor(img, original_pts, reprojec_pts, idx):
    r,g = (0,0,255),(0,255,0)

    for i in range(original_pts.shape[0]):
        # 原始角点
        x0, y0 = int(round(original_pts[i,0])), int(round(original_pts[i,1]))
        cv2.circle(img, (x0, y0), 10, g, 2, lineType=cv2.LINE_AA)
        # 重投影角点
        x1, y1 = int(round(reprojec_pts[i,0])), int(round(reprojec_pts[i,1]))
        cv2.circle(img, (x1, y1), 10, r, 2, lineType=cv2.LINE_AA)
    cv2.imwrite('./reproject_result/'+ str(idx+1)+'.jpg', img)


# 结果评估与可视化
def evaluate(mtx, dist, rvecs, tvecs, world, cam, num):
    # 计算重投影误差
    CalcReprojError(num, world, cam, mtx, dist, rvecs, tvecs)
    # # 可视化相机位姿(棋盘静止)
    show_cam_pose(rvecs, tvecs)
    # # 可视化棋盘位姿(相机静止)
    show_chessboard_pose(rvecs, tvecs)
    plt.show()




# 读取内外参数
def ReadCalibrateParam():
    mtx = np.load('./param/mtx.npy')
    dist = np.load('./param/dist.npy')
    rvecs = np.load('./param/rvecs.npy')
    tvecs = np.load('./param/tvecs.npy')
    world = np.load('./param/world.npy')
    cam = np.load('./param/cam.npy')
    return mtx, dist, rvecs, tvecs, world, cam



if __name__ == '__main__':
    root = './0/'
    ratio = 1 # 4000 / 1920    # 图像缩放比例
    N = 20
    w, h, len = 11, 8, 25  # 标定板规格(内角点个数(11x8)); len是角点间距25mm
    # 标定
    # ret, mtx, dist, rvecs, tvecs = CamCalibrate(ratio, w, h, N, root)
    # 读取标定参数
    mtx, dist, rvecs, tvecs, world, cam = ReadCalibrateParam()
    print("内参矩阵 mtx:\n", mtx)    # 内参数矩阵
    print("畸变系数 dist:\n", dist)   # 畸变系数 
    # 结果评估与可视化
    evaluate(mtx, dist, rvecs, tvecs, world, cam, N)


