#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/8 下午 06:45
# @Author : Li, Yun-Fang
# @File : problem_3.py
# @Software: PyCharm

import numpy as np


def solve():
    X = np.array([[12, 6], [6, 12]])
    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    H = np.dot(X, np.linalg.solve(XtX, Xt))

    print("Case 1")
    print(H)

    X = np.array([[12, 3], [6, 6]])
    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    H = np.dot(X, np.linalg.solve(XtX, Xt))

    print("Case 2")
    print(H)


if __name__ == '__main__':
    solve()
