#!/usr/bin/env python
# coding: utf-8

# In[41]:


#!/usr/local/bin/python

import sys
import numpy as np
import tensorflow as tf
import os
import pickle
from scipy.stats import mode
import warnings

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

import preprocess_v1 as preprocess


# In[52]:


# Create model
def multilayer_perceptron(_X, _weights, _biases):
    # Hidden layer with RELU activation
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), 0.8)
    # Hidden layer with sigmoid activation
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])), 0.8)
    # layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    # layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
    # layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
    return tf.nn.softmax(tf.matmul(layer_2, _weights['out']) + _biases['out'])
    
    
################    Data Loading and Plotting    ########################
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def indices(a, func):
    """Finds elements of the matrix that correspond to a function"""
    return [i for (i, val) in enumerate(a) if func(val)]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load():
    
    n_classes = 33 # Number of classes in bird data
    parametersFileDir = "parameters_mfcc_2.pkl"
    
    
    # Network Parameters
    n_hidden_1 = 256  # 1st layer num features
    #n_hidden_2 = 256  # 2nd layer num features
    # n_hidden_3 = 256 # 3rd layer num features
    # n_hidden_4 = 256
    # n_hidden_5 = 128
    n_input = 585  # input dimensionality

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    #print("Loading saved Weights ...")
    file_ID = parametersFileDir
    f = open(file_ID, "rb")
    W = pickle.load(f)
    b = pickle.load(f)

    # print "b1 = ", b['b1']
    # print "b2 = ", b['b2']
    # print "b3 = ", b['out']

    weights = {
        'h1': tf.Variable(W['h1']),
        'h2': tf.Variable(W['h2']),
        # 'h3': tf.Variable(W['h3']),
        # 'h4': tf.Variable(W['h4']),
        # 'h5': tf.Variable(W['h5']),
        'out': tf.Variable(W['out'])
    }

    biases = {
        'b1': tf.Variable(b['b1']),
        'b2': tf.Variable(b['b2']),
        # 'b3': tf.Variable(b['b3']),
        # 'b4': tf.Variable(b['b4']),
        # 'b5': tf.Variable(b['b5']),
        'out': tf.Variable(b['out'])
    }
    # print type(b['b1'])
    # print type(biases['b1'])

    f.close()


    pred = multilayer_perceptron(x, weights, biases)

    #print("Testing the Neural Network")
    init = tf.initialize_all_variables()
    
    
    with tf.Session() as sess:
        sess.run(init)
        file_specified = '/tmp/test.mfcc'
        example = np.loadtxt(file_specified)
        i = 0
        rows, cols = example.shape
        context = np.zeros((rows - 14, 15 * cols))  # 15 contextual frames
        while i <= (rows - 15):
            ex = example[i:i + 15, :].ravel()
            ex = np.reshape(ex, (1, ex.shape[0]))
            context[i:i + 1, :] = ex
            i += 1
        # see = tf.argmax(pred, 1)
        see = tf.reduce_sum(pred, 0)

        confidence_matrix = softmax(see.eval({x: context}))
        
        #print(confidence_matrix)
        
        confidence_matrix = confidence_matrix * 100
        

        list1 = ['black_throated_tit', 'blackandyellow_grosbeak', 'blackcrested_tit', 'chestnutcrowned_laughingthrush',
                 'darksided_flycatcher', 'eurasian_treecreeper',  'golden_bushrobin', 'great_barbet', 
                 'grey_bellied_cuckoo', 'grey_bushchat', 'greyheaded_canary_flycatcher', 'greyhooded_warbler', 
                 'greywinged_blackbird', 'himalayan_monal', 'humes_warbler', 'large_hawkcuckoo', 
                 'largebilled_crow', 'lesser_cuckoo-GHNP', 'orangeflanked_bushrobin', 'oriental_cuckoo', 
                 'palerumped_warbler', 'redbilled_chough', 'rock_bunting', 'rufous_gorgetted_flycatcher',
                 'rufousbellied_niltava', 'spotted_nutcracker', 'streaked_laughingthrush', 'variegated_laughingthrush' ,
                 'western_tragopan', 'whistlers_warbler', 'whitebrowed_fulvetta' , 'whitecheeked_nuthatch',
                'yellowbellied_fantail']

        #Top Three Labels
        res = np.asarray(confidence_matrix)

        result = {}
        
        for i in range(3):
            rank = {}
            product = np.argmax(res)
            tmp = str(res[product])
            rank[list1[product]] = tmp[:4]
            result[i] = rank
            res[product] = 0
            
        #print(result)
        return result
     
        
        

def predict(filename):
    preprocess.preprocess(filename)
    result = load()
    #print(result)
    return result
    
    
#predict("sample.wav")


# In[ ]:




