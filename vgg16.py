from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, GlobalMaxPool2D, AveragePooling2D
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
from keras import backend as K
from keras.engine import get_source_inputs

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, int_pooling='max', classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The "weigths" argument should be either "None" or "imagenet"')
    
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using "weights" as imagenet with "include_top" as True, "classes" should be 1000')
    
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=48, data_format=K.image_data_format(), include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # Block 1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    if int_pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    else:
        x = MaxPool2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    if int_pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block2_pool')(x)
    else:
        x = MaxPool2D((2,2), strides=(2,2), name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    if int_pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block3_pool')(x)
    else:
        x = MaxPool2D((2,2), strides=(2,2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    if int_pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block4_pool')(x)
    else:
        x = MaxPool2D((2,2), strides=(2,2), name='block4_pool')(x)
    
    # Block 5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
    if int_pooling == 'avg':
        x = AveragePooling2D((2,2), strides=(2,2), name='block5_pool')(x)
    else:
        x = MaxPool2D((2,2), strides=(2,2), name='block5_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPool2D()(x)

    # check input_shape predecessors
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='vgg16')
    
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_dir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_dir='models')
        
        model.load_weights(weights_path)
    
    return model