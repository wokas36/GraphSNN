import keras.backend as K
import tensorflow as tf
import numpy as np

def gat_graph_conv(x, adj, eps, kernel):
        
        v = eps*tf.diag_part(adj)
        mask = tf.diag(tf.ones_like(v))
        adj = mask*tf.diag(v) + (1. - mask)*adj
        
        y1 = K.dot(adj, x)
        
        conv_op_y1 = tf.split(y1, 1, axis=0)
        conv_op_y1 = K.concatenate(conv_op_y1, axis=1)
        conv_op_y1 = K.dot(conv_op_y1, kernel)
        
        conv_out = conv_op_y1
        
        return conv_out