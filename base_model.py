#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from layers import *
from os.path import join as pjoin
import tensorflow as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    Build model architecture.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]

    # Ko: temperal dimention after GLU and spatial gated block, as well as kernel size of convolutional unified layer.
    Ko = n_his
    x = spatial_gated_block(x, Ks, Kt, blocks, keep_prob)
    Ko -= 2 * (Ks - 1)

    if Ko > 1:
        x1 = convolution_unified_layer(x, Ko, 'convolution_unified_layer')
        y = output_layer(x1, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
