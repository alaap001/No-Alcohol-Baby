# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:30:58 2019

@author: wwech
"""


## libraries for KERAS CNN MODEL
from tensorflow.keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Activation, \
                      BatchNormalization ,GlobalAveragePooling2D, concatenate, AveragePooling2D, Input
from tensorflow.keras.models import Model 

class DenseNet:
    
    """
        dense_block_size = 4
        layers_in_block = 4
        growth_rate = 12
        classes = 224
        i_shape = (128,128,3)
        model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block,i_shape)
        model.summary()
    
    """
    
    def __init__(self):
        pass


    def conv_layer(self,conv_x, filters):
        """
        Creates a convolution block consisting of BN-ReLU-Conv.
    
        """
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)
        conv_x = Conv2D(filters, (3, 3), padding='same', use_bias=False)(conv_x)
        conv_x = Dropout(0.3)(conv_x)
    
        return conv_x

  
    def dense_block(self,block_x, filters, growth_rate, layers_in_block):
        """
        Creates a dense block and concatenates inputs
        """
        for i in range(layers_in_block):
            each_layer = self.conv_layer(block_x, growth_rate)
            block_x = concatenate([block_x, each_layer], axis=-1)
            filters += growth_rate
    
        return block_x, filters
      
    def transition_block(self,trans_x, tran_filters):
        """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as downsampling.
        """
        trans_x = BatchNormalization()(trans_x)

        trans_x = Activation('relu')(trans_x)
        trans_x = Conv2D(tran_filters, (1, 1), padding='same', use_bias=False)(trans_x)
        trans_x = AveragePooling2D((2, 2),padding = 'same', strides=(2, 2))(trans_x)
    
        return trans_x, tran_filters
    
    
    def dense_net(self,filters, growth_rate, classes, dense_block_size, layers_in_block, i_shape):
        """
        Creating a DenseNet
        
        Arguments:
            i_shape  : shape of the input images. E.g. (28,28,1) for MNIST    
            classes : number of classes
            filter : number of filters
            dense_blocks_size : amount of dense blocks that will be created    
            layers_in_block : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
            growth_rate  : number of filters to add per dense block (default: 12)        
        Returns:
            Model        : A Keras model instance
        """
      
      
        input_img = Input(shape=i_shape)
        # create the initial conv layer
        
        x = Conv2D(32, (3, 3), padding='same', use_bias=False)(input_img)
        dense_x = BatchNormalization()(x)
        dense_x = Activation('relu')(x)
        dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
        
        #create dense blocks and add transition layers
        for block in range(dense_block_size - 1):
            dense_x, filters = self.dense_block(dense_x, filters, growth_rate, layers_in_block)
            dense_x, filters = self.transition_block(dense_x, filters)
        
        #create last block and perform global average
        dense_x, filters = self.dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x = BatchNormalization()(dense_x)
        dense_x = Activation('relu')(dense_x)
        dense_x = GlobalAveragePooling2D()(dense_x)
    
        output = Dense(classes,activation = 'sigmoid')(dense_x)
    
        return Model(input_img, output)
    
    


