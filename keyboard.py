# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt

#高斯平滑减少图像噪声
img = cv2.imread('lbq.jpg', 0)
img = cv2.GaussianBlur(img, (5, 5), 0)


# 利用自适应阈值函数，求块的高斯分布加权和，二值化类型为"大于最大值"
#然后进行开运算
img_gauss_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15, 2)
kernel = np.ones((1, 1), np.uint8)
img_opening = cv2.morphologyEx(img_gauss_thresh, cv2.MORPH_OPEN, kernel)

# 遍历待测图像，确定白色黑色的范围
height = img_opening.shape[0]
width = img_opening.shape[1]

column_wh = []
column_bl = []
column_wh_max = 0
for i in range(width):
    s = 0
    t = 0
    for j in range(height):
        if img_opening[j][i] == 255:
            s += 1
        if img_opening[j][i] == 0:
            t += 1
    column_wh_max = max(column_wh_max, s)
    column_wh.append(s)
    column_bl.append(t)

row_wh = []
row_bl = []
row_wh_max = 0
for i in range(height):
    s = 0
    t = 0
    for j in range(width):
        if img_opening[i][j] == 255:
            s += 1
        if img_opening[i][j] == 0:
            t += 1
    row_wh_max = max(row_wh_max, s)
    row_wh.append(s)
    row_bl.append(t)



#寻找行和列的结束值
def find_column_end(start_):
    end_ = start_+1
    for m in range(start_+1, width-1):
        if (column_wh[m]) > (0.98 * column_wh_max):
            end_ = m
            break
    return end_

def find_row_end(start_):
    end_ = start_+1
    for m in range(start_+1, width-1):
        if (row_wh[m]) > (0.95 * row_wh_max):
            end_ = m
            break
    return end_

#计算
#黑色和白色距离
mindis = 7
n = 1
start = 1
end = 2
column_st = []
column_end = []
count = 0
while n < width - 2:
    n += 1
    if(column_bl[n]) > (0.15 * column_wh_max):
        start = n
        end = find_column_end(start)
        n = end
        if end-start > mindis:
            count += 1
            column_st.append(start)
            column_end.append(end)

n = 1
start = 1
end = 2
row_st = []
row_end = []
while n < height -2:
    n += 1
    if(row_bl[n]) > (0.1 * row_wh_max):
        start = n
        end = find_row_end(start)
        n = end
        if end-start > 50:
            row_st.append(start)
            row_end.append(end)

#根据标准化的图像计算磨损度
img_st = [8054, 9742, 9181, 9103, 9023, 9821, 10389, 9237, 10356, 9994, 10653, 9768, 10678, 10067, 8326, 8631]
black_dot = []
cnt = 0
denc = 20
for i in range(count):
    for j in range(count):
        img_1 = cv2.imread('unti.jpg', 0)
        img_cut = img_gauss_thresh[row_st[i]+denc:row_end[i]-denc, column_st[j]+denc:column_end[j]-denc]
        temp = 0
        for m in range(row_end[i] - row_st[i] - denc*2):
            for n in range(column_end[j] - column_st[j] - denc*2):
                if img_cut[m][n] == 0:
                    temp += 1
        black_dot.append(temp)

        cv2.imshow('img_cut', img_cut)
        perc = 1 - temp / img_st[4*i+j]
        text=('The attrition rate of this button is %.2f%%' % (perc * 100))
        cv2.putText(img_1, str(text),(50,120),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1)
        cv2.imshow('img_per', img_1)
        cv2.waitKey(0)