# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
# Copyright (c) 2017 S.Masuda.
# 

import sys
import math
import numpy as np
import tensorflow as tf

sys.dont_write_bytecode = True
from tf_util import Convolution2D
from tf_util import FullyConnected


class Convolution2D :
    def __init__(self,in_ch,out_ch,ksize,stride) :
        self.stride = stride
        self.w = tf.Variable( tf.truncated_normal([ksize,ksize,in_ch,out_ch]) )
        self.b = tf.Variable( tf.truncated_normal([out_ch]) )
    
    def linear(self,x) :
        return tf.nn.conv2d( x, self.w, strides=[1, self.stride, self.stride, 1], padding="SAME" ) + self.b

    def relu(self,x) :
        return tf.nn.relu( self.linear(x) )
    
class FullyConnected :
    def __init__(self,n_in,n_out) :
        self.w = tf.Variable( tf.truncated_normal([n_in,n_out],stddev=0.0001) )
        self.b = tf.Variable( tf.truncated_normal([n_out],stddev=0.0001) )
        
    def linear(self,x) :
        return tf.matmul(x,self.w) + self.b 
    
    def relu(self,x) :
        return tf.nn.relu( self.linear(x) )


class SpatialSoftmax :
    def __init__( self, shape = [-1,80,80,16] ) :
        self.shape = shape
        # @JP: 特徴座標抽出用の行列生成
        np_fe_x = np.zeros( (shape[1]*shape[2]*shape[3],shape[3]*2) ,dtype=np.float32 )
        np_fe_y = np.zeros( (shape[1]*shape[2]*shape[3],shape[3]*2) ,dtype=np.float32 )
            
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):    
                for t in xrange(shape[3]):   
                    np_fe_x[ y*(shape[2]*shape[3]) + x*shape[3] + t ][t*2+1] = x +1 
                    np_fe_y[ y*(shape[2]*shape[3]) + x*shape[3] + t ][t*2  ] = y +1
        
        self.fe_x = tf.constant(np_fe_x)
        self.fe_y = tf.constant(np_fe_y)

    def act(self, x) :
        # convert to [-1, ch, height, width]
        trans = tf.transpose(x,perm=[0, 3, 1, 2])  
        
        # tf.nn.softmaxをつかうために、一度 [batch_size * ch, height*width ] のshapeに変換
        dist_to_batch = tf.reshape(  trans, [-1, self.shape[1]*self.shape[2]] )
        
        spatial_softmax = tf.nn.softmax( dist_to_batch )
        
        batch_to_dist = tf.reshape( spatial_softmax, [-1, self.shape[3], self.shape[1], self.shape[2]] )
        
        # convert to [-1, height, width , ch ]
        distributed = tf.transpose( batch_to_dist, perm=[0,2,3,1] )
        softmax_out = tf.reshape( distributed , [-1, self.shape[1]*self.shape[2]*self.shape[3]]  )
        
        dy = float(self.shape[1])
        dx = float(self.shape[2])
        # feature_points
        # TODO
        return (tf.matmul( softmax_out, self.fe_y ) ) + (tf.matmul( softmax_out, self.fe_x ) )



class DeepSpatialAutoEncoder:
    def __init__(
            self,
            input_shape = [-1,240,240,3],
            reconstruction_shape = [-1,60,60,1],
            filter_chs = [64,32,16],
            filter_sizes = [7,5,5],
            filter_strides = [2,1,1]
            ):
        self.filter_num = len(filter_chs)
        if any(len(lst) != self.filter_num for lst in [filter_chs,filter_sizes,filter_strides]):
            raise NameError("size error.")
        
        
        self.input_shape = input_shape
        self.reconstruction_shape = reconstruction_shape

        self.filter_chs = filter_chs
        self.filter_sizes = filter_sizes
        self.filter_strides = filter_strides

        self.convs = []
        self.shapes = [] # shape[0] -> conv1 -> shape[1] -> conv2 -> shape[2] -> ...
        self.shapes.append(input_shape)
        
        for i,(fch,fsize,fstride) in enumerate(zip(filter_chs,filter_sizes,filter_strides)) :
            in_shape = self.shapes[-1]
            conv = Convolution2D( in_ch=in_shape[3],out_ch=fch,ksize=fsize,stride=fstride)
            self.convs.append(conv)
            out_height = int( math.ceil(float(self.shapes[-1][1]) / float(fstride)) )
            out_width = int( math.ceil(float(self.shapes[-1][2]) / float(fstride)) )
            self.shapes.append([-1,out_height,out_width,fch])
        # endof for

       
        self.spatial_softmax = SpatialSoftmax( self.shapes[-1] )
        self.fully_connected = FullyConnected( self.filter_chs[-1]*2, reconstruction_shape[1]*reconstruction_shape[2] )

    def add_encode(self, x) :
        i_input = x
        for i in range(len(self.convs)) :
            h = self.convs[i].relu( x=i_input )
            i_input = h
        # endof for
        h = self.spatial_softmax.act( i_input )
        return h

    def add_decode(self, x) :
        return self.fully_connected.linear(x)

    def add_train_without_gslow(self, x) :
        encoded = self.add_encode(x)
        decoded = self.add_decode(encoded)
        decoded_img = tf.reshape(decoded, [-1, self.reconstruction_shape[1],self.reconstruction_shape[2],1]) 

        sv = tf.image.rgb_to_grayscale( x )
        sv = tf.image.resize_images( sv, [self.reconstruction_shape[1],self.reconstruction_shape[2]],tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        sv_flatten = tf.reshape(sv, [-1, self.reconstruction_shape[1]*self.reconstruction_shape[2]])

        loss = tf.reduce_sum( tf.square( sv_flatten - decoded ) )
        
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(0.0005, step, 1, 0.9999)
        optimizer = tf.train.AdamOptimizer(rate)
        targs = []
        for i in range(len(self.convs)) :
            targs.append(self.convs[i].w)
            targs.append(self.convs[i].b)
        targs.append(self.fully_connected.w)
        targs.append(self.fully_connected.b)

        grads = optimizer.compute_gradients( loss, targs )                 
        train_step = optimizer.apply_gradients(grads)
        return train_step,loss,decoded_img,sv

    
    def add_train(self, x, x_next) :
        encoded = self.add_encode(x)
        decoded = self.add_decode(encoded)
        decoded_img = tf.reshape(decoded, [-1, self.reconstruction_shape[1],self.reconstruction_shape[2],1]) 

        encoded_next = self.add_encode(x_next)

        i_downsamp = tf.image.rgb_to_grayscale( x )
        i_downsamp = tf.image.resize_images( i_downsamp, [self.reconstruction_shape[1],self.reconstruction_shape[2]],tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        i_downsamp_flatten = tf.reshape(i_downsamp, [-1, self.reconstruction_shape[1]*self.reconstruction_shape[2]])

        #gslow = tf.square( (encoded_next-encoded) - (encoded - encoded_prev) )
        gslow = tf.square( (encoded_next-encoded) )
        loss = tf.reduce_sum( tf.square( i_downsamp_flatten - decoded ) ) + tf.reduce_sum( gslow )
        
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(0.0005, step, 1, 0.9999)
        optimizer = tf.train.AdamOptimizer(rate)
        targs = []
        for i in range(len(self.convs)) :
            targs.append(self.convs[i].w)
            targs.append(self.convs[i].b)
        targs.append(self.fully_connected.w)
        targs.append(self.fully_connected.b)

        grads = optimizer.compute_gradients( loss, targs )                 
        train_step = optimizer.apply_gradients(grads)
        return train_step,loss,decoded_img,i_downsamp


