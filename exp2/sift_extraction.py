
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np   
import os
import cv2
from tqdm import tqdm, trange
import pydot

import sys;sys.path.append('../')
import utils





# 提取图像SIFT特征点及描述子并保存为文本文件
def extraSIFT2txt(src, name, root='./'):
    # 转化为灰度图
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # 灰度图另存为
    img_name = 'tmp.pgm'
    cv2.imwrite(img_name, src)
    dst_name = name + '.sift'
    
    cmd_param = ' --edge-thresh 5 --peak-thresh 10'
    command = 'sift ' + img_name + ' --output=' + root + dst_name + cmd_param 
    # 执行命令行命令
    os.system(command)
    print('图像:' + name + ' 特征点提取完毕')



# 读取文本文件中的SIFT特征并转化为numpy数组
def txt2SIFT(name, root='./c/'):
    cors = open(root+name+'.sift','r').readlines()
    x, y, scale, direction, desc = [], [], [], [], []
    for cor in cors:
        cor = cor.split(' ')[:-1] # 去除掉换行符
        x.append(float(cor[0]))          # 坐标x
        y.append(float(cor[1]))          # 坐标y
        scale.append(float(cor[2]))      # 尺度
        direction.append(float(cor[3]))  # 方向
        desc.append([int(i) for i in cor[4:]]) # 描述子
    cors_info = np.array([x,y,scale,direction]).T
    desc = np.array(desc)
    return cors_info, desc


# SIFT角点提取
def extraSIFTfromImg(img, name):
    extraSIFT2txt(img, name)
    return txt2SIFT(name)


# 描述子单向匹配
def singleMatch(norm_desc1, norm_desc2):
    match_seq = []  # 匹配序列
    # 计算余弦距离
    sim_matrix = norm_desc2 @ norm_desc1.T
    sim_matrix = sim_matrix.T # 相似矩阵
    # 可视化相似矩阵
    # plt.imshow(sim_matrix)
    # plt.show()
    # 从大到小顺序排列
    sim_idx = np.argsort(sim_matrix, 1)[:,::-1]
    for i in range(sim_idx.shape[0]):
        top1_idx, top2_idx = sim_idx[i,0], sim_idx[i,1]
        top1_val = sim_matrix[i,top1_idx]
        top2_val = sim_matrix[i,top2_idx]
        # 最近邻角度/第二近邻角度<阈值 ? 是匹配点 : 舍弃
        if np.arccos(top1_val) < 0.6 * np.arccos(top2_val):
            match_seq.append([i, top1_idx])
        else:
            match_seq.append([i, -1])

    return np.array(match_seq)


# 描述子双向匹配
def doubleMatch(desc1, desc2):
    # 标准归一化
    norm_desc1 = norm(desc1)
    norm_desc2 = norm(desc2)
    # 双向匹配
    matches_12 = singleMatch(norm_desc1, norm_desc2)
    matches_21 = singleMatch(norm_desc2, norm_desc1)
    matcher = []
    for i in range(matches_12.shape[0]):
        if(matches_12[i,1] != -1): # 排除掉无匹配的点
            # 若双向匹配是对称的，才是合格的匹配点
            if matches_21[matches_12[i,1], 1] == matches_12[i,0]:
                matcher.append([matches_12[i,0], matches_12[i,1]])
    return np.array(matcher)



# Harris角点提取
def extraHarrisfromImg(img, threshold):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    img = np.float32(img)
    # opencv函数计算响应值
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    plt.imshow(dst)
    plt.show()
    # 实际阈值
    threshold = threshold*dst.max()
    # 大于阈值的设为角点
    cor_axis = np.array(np.where(dst > threshold)).T[:,[1,0]]
    # 返回角点坐标
    return cor_axis





# 可视化SIFT匹配对(从文件夹中读取)
def drawMatchesfromDir(data_root, result_root, name1, name2):
    img1,_,_ = utils.auto_reshape(cv2.imread(data_root + name1 + '.jpg'), 1080)
    img2,_,_ = utils.auto_reshape(cv2.imread(data_root + name2 + '.jpg'), 1080)

    sift_cors1, sift_desc1 = txt2SIFT(name1, result_root)
    sift_cors2, sift_desc2 = txt2SIFT(name2, result_root)

    matches = doubleMatch(sift_desc1, sift_desc2)
    utils.drawMatches(img1, img2, sift_cors1, sift_cors2, matches)





# 批量提取图像SIFT特征
def extractBatchSIFTs(data_root, result_root):
    img_path, img_name = utils.read_img_files(data_root)
    for i in range(len(img_path)):
        img = cv2.imread(img_path[i])
        img, _, _ = utils.auto_reshape(img, 1080)
        extraSIFT2txt(img, img_name[i], result_root)



# 批量进行图像SIFT特征匹配
def matchBatchImgs(data_root, result_root, extraSIFT=False):
    if extraSIFT:
        extractBatchSIFTs('./datasets/', './cor_info/')
    img_path, img_name = utils.read_img_files(data_root)
    img_nums = len(img_name)
    # 获取该批图像的所有描述子:
    desc_list = []
    for i in range(img_nums):
        _, desc = txt2SIFT(img_name[i], result_root)
        desc_list.append(desc)
    # 两两进行匹配：
    match_matrix = np.zeros((img_nums, img_nums))
    for i in tqdm(range(img_nums-1), desc='Processing'+str(i)):
        j=i+1
        for j in tqdm(range(i+1, img_nums), desc='Processing'+str(j)):
            matches = doubleMatch(desc_list[i], desc_list[j])
            match_matrix[i,j] = matches.shape[0]
    # 可视化匹配矩阵
    origin_matrix = match_matrix
    plt.imshow(match_matrix)
    plt.show() 
    # 下面这段代码删除无匹配点的矩阵行
    zero_rows = []
    for i in range(img_nums):
        if(sum(match_matrix[i,:]) == 0):
            zero_rows.append(i)
    match_matrix = np.delete(match_matrix, zero_rows, axis=0)
    # 匹配索引矩阵:
    match_idx_matrix = np.argsort(match_matrix, 1)[:,::-1]
    match_sort_matrix = np.sort(match_matrix, 1)[:,::-1]
    # 将那些0匹配的图像对设为-1，不再考虑
    match_idx_matrix[match_sort_matrix==0] = -1
    # 输出匹配图像的名称(无向图)
    match_img_name = {}
    for i in range(match_idx_matrix.shape[0]):
        names = []
        for idx in match_idx_matrix[i,:]:
            if idx == -1:break
            names.append(img_name[idx])
        match_img_name[img_name[i]] = names
    print(match_img_name)
    return origin_matrix# match_matrix   # 无向图邻接矩阵



# 匹配与当前图像相似的帧
def matchImgViz(data_root, result_root, downsample=False):
    img_path, img_name = utils.read_img_files(data_root)
    small_data_root = 'D:/YHT/学习/大三下/computer_vision/exp2/datasets_small/'
    if(downsample):
        for i in range(len(img_path)):
            img = cv2.imread(img_path[i])
            img,_,_ = utils.auto_reshape(img, 200)
            cv2.imwrite('./datasets_small/' + img_name[i]+'.png', img)
            print(small_data_root + img_name[i]+'.jpg')
    
    match_matrix = np.load('match_matrix.npy')
    g = pydot.Dot(graph_type='graph')
    for i in range(match_matrix.shape[0]-1):
        g.add_node(pydot.Node(str(i), shape='rectangle', image=small_data_root + img_name[i]+'.png'))
        for j in range(i+1,match_matrix.shape[0]):
            if(match_matrix[i,j]>1):
                print(small_data_root + img_name[i]+'.png')
                g.add_node(pydot.Node(str(j), shape='rectangle', image=small_data_root + img_name[i]+'.png'))
                g.add_edge(pydot.Edge(str(i), str(j)))
    g.write_png('graph.jpg')






# 只匹配与当前帧最相似的帧(匹配点最多)(用于拍摄时序还原)
def matchMaxImgViz(data_root, result_root, downsample=False):
    img_path, img_name = utils.read_img_files(data_root)
    small_data_root = 'D:/YHT/学习/大三下/computer_vision/exp2/datasets_small/'
    if(downsample):
        for i in range(len(img_path)):
            img = cv2.imread(img_path[i])
            img,_,_ = utils.auto_reshape(img, 200)
            cv2.imwrite('./datasets_small/' + img_name[i]+'.png', img)
            print(small_data_root + img_name[i]+'.jpg')
    
    match_matrix = np.load('match_matrix.npy')
    match_matrix += match_matrix.T
    mark = np.zeros(len(img_path))
    g = pydot.Dot(graph_type='graph')
    i = 0
    while(sum(mark)<len(img_path)):
        print(i)
        mark[i] = 1
        max_id = -1
        most_matches = 0
        g.add_node(pydot.Node(str(i), shape='rectangle', image=small_data_root + img_name[i]+'.png'))
        for j in range(len(img_path)):
            if(match_matrix[i,j]>most_matches and mark[j]==0):
                most_matches = match_matrix[i,j]
                max_id = j
        g.add_edge(pydot.Edge(str(i), str(max_id)))
        i = max_id
    g.write_png('graph.jpg')





if __name__ == '__main__':
    data_root, result_root = './datasets/', './cor_info/'
    # data_root, result_root = './d/', './c/'
    # 批量提取sift特征
    # extractBatchSIFTs(data_root, result_root)
    img_path, img_name = utils.read_img_files(data_root)
    # img1 = cv2.imread(img_path[0])
    # img1,_,_ = utils.auto_reshape(img1, 1080)

    # harris_cors = extraHarrisfromImg(img1, 0.5)
    # print(harris_cors.shape)
    # drawCor(img1, harris_cors)

    # sift_cors, desc = extraSIFTfromImg(img1, img_name[0])
    # print(sift_cors.shape)
    # drawCor(img1, sift_cors)





    # 任给两张图像路径可视化sift匹配对
    drawMatchesfromDir(data_root, result_root, img_name[27], img_name[26])
    # extractBatchSIFTs(data_root, result_root)
    # match_matrix = matchBatchImgs(data_root, result_root)
    # np.save('match_matrix.npy', match_matrix)
    # matchMaxImgViz(data_root, result_root)






# activate E:\environments\media_pipe