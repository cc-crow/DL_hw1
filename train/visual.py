# coding=gbk

import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

param = np.load('/home/leijingshi/DL/homework1/train/parameter/param.npy', allow_pickle=True)

def img_plot(param,shape,filename,ncol):
    c = param.shape[1]
    nrow = int(c/ncol)+1
    plt.figure(dpi=100)
    for i in range(c):
        img = param[:,i]
        img = img.reshape(*shape)
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
        img = Image.fromarray(np.uint8(img))
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    #plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(filename)

img_plot(param[0][:,:30],(28,28),'/home/leijingshi/DL/homework1/train/figures/weight1.jpg',5)
img_plot(param[2],(16,32),'/home/leijingshi/DL/homework1/train/figures/weight2.jpg',5)