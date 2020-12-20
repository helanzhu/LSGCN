#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

def spatial_gated_block(x, Ks, Kt, channels, keep_prob):
    '''
    LSGCN spatial gated block
    :param x: tensor, batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of graph convolution.
    :param Kt: int, kernel size of GLU convolution.
    :param channels: list, channels of every convolutional layers.
    :param keep_prob: placeholder, probability of dropout.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    c_si, c_t, c_oo = channels
    _, Block_T, Block_n, _ = x.get_shape().as_list()
    with tf.variable_scope('model_in'):
        x_s = GLU_conv_layer(x, Kt, c_si, c_t)
    with tf.variable_scope('spatial_gated_block_in'):
        x_s_a = spatial_attention_cosAtt(x_s)
        x_s_b = spatial_conv_layer(x_s, Ks, c_t, c_t)
        x_t = x_s_a*tf.nn.sigmoid(x_s_b)
    with tf.variable_scope('spatial_gated_block_out'):
        x_o = GLU_conv_layer(x_t, Kt, c_t, c_oo)
    x_ln = layer_norm(x_o, 'layer_norm')
    return tf.nn.dropout(x_ln, keep_prob)


def GLU_conv_layer(x, Kt, c_in, c_out):
    '''
    GLU convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size of temporal convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()
    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]  #  Kt-1:T

    wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
    bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
    x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
    return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])


def spatial_attention_cosAtt(x_input):
    '''
    Spatial attention cosAtt layer without dimention change.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    x = x_input
    _batch_size_, _T_, _n_, _feature_ = x.get_shape().as_list()
    norm2 = tf.norm(x, ord=2, axis=-1, keepdims=True)
    x_result= tf.matmul(x, tf.transpose(x, [0, 1, 3, 2]))
    norm2_result = tf.matmul(norm2, tf.transpose(norm2, [0, 1, 3, 2]))
    attention_beta = tf.get_variable('attention_beta', shape=[1, 1, _n_, _n_], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(attention_beta))

    cos = tf.divide(x_result, norm2_result+1e-7)
    cos_ = tf.multiply(attention_beta, cos)
    P = tf.nn.sigmoid(cos_)
    output = tf.matmul(P, x)
    return output


def spatial_conv_layer(x, Ks, c_in, c_out):
    '''
    Spatial graph convolution layer.
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Ks: int, kernel size of spatial convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, time_step, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        # bottleneck down-sampling
        w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
    # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    # x_g -> [batch_size, time_step, n_route, c_out]
    x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
    return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)


def gconv(x, theta, Ks, c_in, c_out):
    '''
    Spectral-based graph convolution function.
    :param x: tensor, [batch_size, n_route, c_in].
    :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
    :param Ks: int, kernel size of graph convolution.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :return: tensor, [batch_size, n_route, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def layer_norm(x, scope):
    '''
    Layer normalization function.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, time_step, n_route, channel].
    '''
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)

    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x


def convolution_unified_layer(x, T, scope):
    '''
    convolution_unified_layer: map several time steps to one.
    :param x: tensor, [batch_size, time_step, n_route, channel].
    :param T: int, kernel size of convolution, which map time step to one.
    :param scope: str, variable scope.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps time step to one.
    with tf.variable_scope(f'{scope}_in'):
        x_i = GLU_conv_layer(x, T, channel, channel)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    return x_ln


def output_layer(x_ln, scope):
    '''
    convolution_unified_layer: map several time steps to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    _, _, n, channel = x_ln.get_shape().as_list()
    with tf.variable_scope(f'{scope}_out'):
        x_o = activation_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    # Convert multi-channels to one.
    x_fc3 = fully_con_layer(x_o, n, channel, scope)
    return x_fc3


def activation_layer(x, Kt, c_in, c_out, act_func='relu'):
    '''
    :param x: tensor, [batch_size, time_step, n_route, c_in].
    :param Kt: int, kernel size.
    :param c_in: int, size of input channel.
    :param c_out: int, size of output channel.
    :param act_func: str, activation function.
    :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        # if the size of input channel is less than the output,
        # padding x to the same size of output channel.
        # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    else:
        x_input = x

    # keep the original input for residual connection.
    x_input = x_input[:, Kt - 1:T, :, :]  #  Kt-1:T
    wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
    bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
    x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
    if act_func == 'linear':
        return x_conv
    elif act_func == 'sigmoid':
        return tf.nn.sigmoid(x_conv)
    elif act_func == 'relu':
        return tf.nn.relu(x_conv + x_input)
    else:
        raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer: maps multi-channels to one.
    :param x: tensor, [batch_size, 1, n_route, channel].
    :param n: int, number of route / size of graph.
    :param channel: channel size of input x.
    :param scope: str, variable scope.
    :return: tensor, [batch_size, 1, n_route, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

