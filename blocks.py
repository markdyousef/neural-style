from keras.layers import *

def conv_block(x, filters, size=(3,3), strides=(2,2), padding='same', act=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x) if act else x

def deconv_block(x, filters, size=(3,3), strides=(2,2)):
    x = DeConv2D(filters, size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def up_block(x, filters, size):
    x = UpSampling2D()(x)
    x = Conv2D(filters, size, padding='same')(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def res_block(ip, filters, index=0):
    x = conv_block(ip, filters, strides=(1,1))
    x = conv_block(x, filters, strides=(1,1), act=False)
    return add([x, ip])

def res_crop_block(ip, filters, index=0):
    x = conv_block(ip, filters, strides=(1,1), padding='valid')
    x = conv_block(x, filters, strides=(1,1), padding='valid', act=False)
    ip = Lambda(lambda x: x[:, 2:-2, 2:-2])(ip)
    return add([x, ip])