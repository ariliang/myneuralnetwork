# -*- coding: UTF-8 -*-
'''
Created on 2020年6月26日

@author: yangjinfeng
'''
import numpy as np

'''
    输入矩阵的规范化
    (x-xbar)/x**2   axis=1
'''


def normalizerData(data):
    miu = np.mean(data, axis=1, keepdims=True)
    sigma = np.mean(data * data, axis=1, keepdims=True)
    return (data - miu) / sigma


'''
安全的矩阵版本sigmoid
    先将值控制在[min, max]范围内
    minn, maxx
    max(minn, min(maxx, x))
'''


def sigmoid(x):
    min = -50
    max = 50
    # 将值控制在[-50, +50]内
    tempz = np.maximum(min, np.minimum(max, x))
    # 计算sigmoid
    return 1 / (1 + np.exp(-tempz))


'''
计算向量范数
    |v| == sqrt(x1**2 + x2**2 + ...)
    |m| == sqrt(|v1|**2 + |v2|**2 + ...)
'''


def calNorm(x):
    return np.linalg.norm(x, ord=None, axis=None, keepdims=False)


if __name__ == '__main__':
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 200, 300], [400, 500, 600]]).T
    print(data)
    print(sigmoid(data))
    print(normalizerData(data))
