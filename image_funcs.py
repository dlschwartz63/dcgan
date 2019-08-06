import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import keras
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from urllib.request import urlretrieve
import zipfile
import matplotlib.pyplot as plt

def get_image(image_path, width, height, mode):
    from PIL import Image
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)
    return np.array(image.convert(mode))
    
def load_imgs(no_images):
    imgpath='/home/ec2-user/DLprojects/data/celeba/img_align_celeba'
    imgs=[]
    for i in os.listdir(imgpath)[0:no_images]:
        imgs.append(get_image(imgpath+'/'+i,28,28,'RGB'))
    return imgs

def plot_imgs(gen,dims):
    noise = np.random.uniform(-1, 1, (dims * dims, 100))
    gen_imgs = gen.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5#from tanh
    fig, axs = plt.subplots(dims, dims,figsize=(2,2))
    cnt = 0
    for i in range(dims):
        for j in range(dims):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return gen_imgs