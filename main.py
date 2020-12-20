#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf
import time
from os.path import join
from preprocess import cheb_poly_approx
from preprocess import data_gen
from preprocess import weight_matrix
from preprocess import scaled_laplacian

from tester import model_test
from trainer import model_train

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_route', type=int, default=228)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--ks', type=int, default=3)
    parser.add_argument('--kt', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--file_path', type=str, default='./datasets/PeMS07/')
    parser.add_argument('--graph', type=str, default='W_228.csv')
    parser.add_argument('--feature', type=str, default='V_228.csv')
    parser.add_argument('--C_i', type=int, default=1)
    parser.add_argument('--C_1', type=int, default=32)
    parser.add_argument('--C_2', type=int, default=64)
    parser.add_argument('--train_days', type=int, default=34)
    parser.add_argument('--validation_days', type=int, default=5)
    parser.add_argument('--test_days', type=int, default=5)
    parser.add_argument('--sum_path', type=str, default='./output/models')#  ./output/tensorboard

    print("# Loading argument...\n")
    args = parser.parse_args()
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    batch_size, epoch = args.batch_size, args.epoch
    Ks, Kt = args.ks, args.kt
    file_path = args.file_path
    GraphFile = args.graph
    DataFile = args.feature
    C_i, C_1, C_2 = args.C_i, args.C_1, args.C_2
    blocks = [C_i, C_1, C_2]
    n_train, n_val, n_test = args.train_days, args.validation_days, args.test_days
    print("""There are %d vertices in dataset %s and %s.\n""" %(n, args.graph, DataFile))

    print("# Preprocessing data...\n")
    W = weight_matrix(join(file_path, GraphFile))
    L = scaled_laplacian(W)
    Lk = cheb_poly_approx(L, Ks, n)
    tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
    PeMS = data_gen(join(file_path, DataFile), (n_train, n_val, n_test), n, n_his + n_pred)

    print("# Training data...\n")
    train_start_time = time.time()
    model_train(PeMS, blocks, args)
    train_end_time = time.time()
    print("# Training time is: " + str(train_end_time - train_start_time))

    print("# Testing data...\n")
    test_start_time = time.time()
    model_test(PeMS, batch_size, n_his, n_pred)
    test_end_time = time.time()
    print("# Testing time is: " + str(test_end_time - test_start_time))


if __name__ == '__main__':
    main()
