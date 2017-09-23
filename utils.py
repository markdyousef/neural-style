import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
import keras.backend as K
from keras.layers import *
from keras import metrics
from keras.preprocessing import image
from glob import glob
from PIL import Image
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from scipy import ndimage
from keras_tqdm import TQDMNotebookCallback
import bcolz as bz

def save_array(fname, arr):
    c = bz.carray(arr, rootdir=fname, mode=w)
    c.flush()

def load_array(fname): return bz.open(fname)[:]

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, class_mode='categorical', target_size=(256,256)):
    return gen.flow_from_directory(path, target_size=target_size, batch_size=batch_size, class_mode=class_mode, shuffle=shuffle)

def get_data(path, target_size=(256,256)):
    batches = get_batches(path, target_size=target_size)
    return np.concatenate([batches.next() for i in range (batches.samples)])