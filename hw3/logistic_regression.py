#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/11/16 下午 01:37
# @Author : Li, Yun-Fang
# @File : logistic_regression.py
# @Software: PyCharm

import numpy as np


data_num = 1000


def preprocess_data(filename):
    data = np.genfromtxt(filename)
    X = data[:, :-1]
    X = np.c_[np.ones(data_num), X]
    Y = data[:, -1].reshape(data_num, 1)

    return X, Y


def cal_pseudo_inverse(X):
    X_t = np.transpose(X)
    X_tX = np.dot(X_t, X)
    X_tX_inverse = np.linalg.inv(X_tX)
    H = np.dot(X_tX_inverse, X_t)

    return H


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def cal_cross_entropy_error(w, X, Y):
    train_Y = np.dot(X, w)
    arr_error = np.log(1 + np.exp(-(Y - train_Y)))
    error = np.sum(arr_error) / data_num

    return error


def solve():
    train_X, train_Y = preprocess_data('train.dat')
    pseudo_inverse = cal_pseudo_inverse(train_X)
    w_lin = np.dot(pseudo_inverse, train_Y)  # linear_regression weight

    ERRORS, eta = [], 0.001

    for i in range(1000):  # repeat the experiment for 1000 times
        w = np.zeros((len(train_X[0]), 1))
        # w = w_lin
        for iteration in range(500):
            # choose one example
            row_rand_array = np.arange(train_X.shape[0])
            np.random.shuffle(row_rand_array)
            x = train_X[row_rand_array[0:1]]
            y = train_Y[row_rand_array[0:1]]
            # update
            w = w + y * eta * np.transpose(sigmoid(-y * np.dot(x, w)) * x)

        # calculate error
        error = cal_cross_entropy_error(w, train_X, train_Y)
        print("Case {0}: {1}".format(i + 1, error))
        ERRORS.append(error)

    print(np.mean(ERRORS))


if __name__ == '__main__':
    solve()
