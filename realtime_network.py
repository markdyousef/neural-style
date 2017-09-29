from keras.layers import *
import tensorflow as tf
from keras.models import Model
import keras.backend as K
from vgg16 import VGG16, preprocess
from PIL import Image
from blocks import conv_block, up_block, res_crop_block, res_block
from layers import ReflectionPadding2d
import numpy as np
from utils import get_batches

def mean_sqr(diff):
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims(K.sqrt(K.mean(diff**2, dims)), 0)

def gram_matrix(x):
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    feat = K.reshape(x, (shape[0], shape[1], shape[2]*shape[3]))
    num = K.batch_dot(feat, K.permute_dimensions(feat, (0, 2, 1)))
    den = K.prod(K.cast(shape[1:], K.floatx()))
    return num/den

weights = [0.1, 0.2, 0.6, 0.1]
def total_loss(x):
    loss = 0
    n = len(style_maps)
    for i in range(n):
        loss += mean_sqr(gram_matrix(x[i+n]) - gram_matrix(style_maps[i])) / 2.
        loss += mean_sqr(x[i]-x[i+n]) * weights[i]
    return loss

class StyleNetwork(Object):
    def __init__(self, input, style_path, **kwargs):
        self.input = input
        self.input_shape = input.shape()
        self.style_path = './data/picasso.jpg'
        self.final_model = None
        self.tf_model = None
        self.loss_model = None
        super(StyleNetwork, self).__init__(**kwargs)
    
    def create_transform_network():
        tf_input = Input(self.input_shape)
        tf_model = ReflectionPadding2d((40,40))(inp)
        tf_model = conv_block(x, 32, size=(9,9), strides=(1,1))
        tf_model = conv_block(x, 64, size=(3,3), strides=(2,2))
        tf_model = conv_block(x, 128, size=(3,3), strides=(2,2))
        for i in range(5): tf_model = res_crop_block(x, 128, i)
        tf_model = up_block(x, 64, size=(3,3))
        tf_model = up_block(x, 32, size=(3,3))
        # x = deconv_block(x, 3, size=(9,9), strides=(1,1))
        tf_model = Conv2D(3, (9,9), activation='tanh', padding='same')(x)
        self.tf_model = Lambda(lambda x: (x+1)*127.5)(x)
        return self.tf_model, tf_input

    def create_loss_network():
        vgg_input = Input(self.input_shape)
        input_tensor=Lambda(preprocess)(vgg_input)
        
        self.vgg_model = VGG16(include_top=False,
                    input_tensor=input_tensor, int_pooling='avg')
        
        return self.vgg_model, vgg_input
    
    def build():
        tf_model, tf_input = create_transform_network()
        # we don't retrain our loss network
        vgg_model, vgg_input = create_loss_network();
        for layer in vgg_model.layers:
            layer.trainable = False
        
        def get_output(model, layer_i):
            return model.get_layer(f'block{layer_i}_conv2').output
        
        # style maps/activations
        shape = self.input_shape
        style_image = Image.open(self.style_path).resize(shape)
        y_s = np.array(style_image)[:shape[0], :shape[1], :shape[2]]

        vgg_content = Model(vgg_input, [get_output(vgg_model, i) for i in [2,3,4,5]])
        style_maps = [K.variable(pred) for pred in vgg_content.predict(np.expand_dims(y_s, 0))]
        yc_act = vgg_content(vgg_input)
        yhat_act = vgg_content(tf_model)

        loss = Lambda(total_loss)(yc_act+yhat_act, style_maps)

        self.final_model = Model([tf_input, vgg_input], loss)
        return self.final_model.compile('adam', 'mae')
    
    def train(x, x1):
        targ = np.zeros((x.shape[0], 1))
        return self.final_model.fit([x, x1], targ, 8, 15)

x_gen = get_batches('./data/train/', target_size=(256,256), class_mode=None)
x = [batch.next() for batch in range(100)]
style_network = StyleNetwork(x)
style_network.build()
style_network.train()