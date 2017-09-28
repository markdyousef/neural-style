from keras.layers import *
import tensorflow as tf
from keras.models import Model
import keras.backend as K
from vgg16 import VGG16, preprocess
from PIL import Image

class ReflectionPadding2d(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2d, self).__init__(**kwargs)
        
    def compute_output_shape(self, shape):
        return (shape[0], shape[1] + 2 * self.padding[0], shape[2] + 2 * self.padding[1], shape[3])
    
    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

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

img = Image.open(fnames[777]);
style = Image.open('./data/picasso.jpg')
style = style.resize(img.size)
y_s = np.array(style)[:input_shape[0], :input_shape[1], :input_shape[2]]; y_s.shape

def create_style_network(input_shape):
    inp = Input(input_shape)
    x = ReflectionPadding2d((40,40))(inp)
    x = conv_block(x, 32, size=(9,9), strides=(1,1))
    x = conv_block(x, 64, size=(3,3), strides=(2,2))
    x = conv_block(x, 128, size=(3,3), strides=(2,2))
    for i in range(5): x = res_crop_block(x, 128, i)
    x = up_block(x, 64, size=(3,3))
    x = up_block(x, 32, size=(3,3))
    # x = deconv_block(x, 3, size=(9,9), strides=(1,1))
    x = Conv2D(3, (9,9), activation='tanh', padding='same')(x)
    out = Lambda(lambda x: (x+1)*127.5)(x)

    vgg_input = Input(input_shape)
    input_tensor=Lambda(preprocess)(vgg_input)
    
    vgg_model = VGG16(include_top=False,
                  input_tensor=input_tensor, int_pooling='avg')

    for layer in vgg_model.layers:
        layer.trainable = False
    
    def get_output(model, layer_i):
        return model.get_layer(f'block{layer_i}_conv2').output

    vgg_content = Model(vgg_input, [get_output(vgg_model, i) for i in [2,3,4,5]])
    style_maps = [K.variable(pred) for pred in vgg_content.predict(np.expand_dims(y_s, 0))]

    yc_act = vgg_content(vgg_input)
    yhat_act = vgg_content(out)

    loss = Lambda(total_loss)(yc_act+yhat_act)

    style_model = Model([inp, vgg_input], loss)

    return style_model.compile('adam', 'mae')

create_style_network((256, 256, 3))