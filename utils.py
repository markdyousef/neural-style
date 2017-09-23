import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
import keras.backend as K
from keras.layers import *
from keras import metrics
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