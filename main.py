from statistics import mode
from unittest.main import MODULE_EXAMPLES
import PIL
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# import torch
import numpy as np
#from torch import nn
from tqdm.auto import tqdm
#from torchvision import transforms
#from torchvision.datasets import CelebA
#from torchvision.utils import make_grid
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import keras
#from torch.distributions import MultivariateNormal
import pandas as pd
#import seaborn as sns 
#import scipy
#from skimage.transform import resize
#from torchvision.models import inception_v3
from PIL import Image 
import math
#!pip install -Uqq ipdb
#import ipdb
import pdb

import warnings
warnings.filterwarnings("ignore")
#from torchvision.models import inception_v3

from disc import *
from gen import *
from utils import *
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# model_pth= 'gdrive/MyDrive/cloud_gan/modeltmp/'
# out_pth = 'gdrive/MyDrive/cloud_gan/outtmp/'
# test_model_pth = 'gdrive/MyDrive/cloud_gan/model128/generator_090.h5'
# data_pth = 'gdrive/MyDrive/cloud_gan/gan_dataset/understanding_cloud_organization/train_images' 
# inception_pth= 'gdrive/MyDrive/cloud_gan/inception_v3_google-1a9a5a14.pth'

model_pth = 'model'
out_pth= 'out'
data_pth= 'dataset/train_images'
inception_pth= 'inception_v3_google-1a9a5a14.pth'
datalist=[]
for img in tqdm(os.listdir(data_pth)):
  datalist.append(img)
print(len(datalist), datalist[0])

# inception_model = inception_v3(pretrained=False)
# inception_model.load_state_dict(torch.load(inception_pth))
# #inception_model.to(device)
# inception_model = inception_model.eval() # Evaluation mode
# inception_model.fc = torch.nn.Identity()
# # gen.eval()
n=3
evalbatch = n*n
n_samples= 1024 #1024
device= 'cpu'
freq= 5
latent_dim = 100
img_shape=(128,128,3)
utils= utils(out_pth, model_pth, freq, img_shape)

checkpoint= None
if checkpoint:
  d_model = keras.models.load(f'checkpoint/disc_{checkpoint}.h5')
  g_model = keras.models.load(f'checkpoint/gen_{checkpoint}.h5')
else:
  d= discriminator()
  g= generator()
  d_model = d(img_shape)
  #print(d_model.summary())
  g_model = g(latent_dim)
  #print("_________________________________________________________________")
  #print(g_model.summary())
gan_model = define_gan(g_model, d_model)
utils.train(data_pth, datalist, g_model, d_model, gan_model, latent_dim, n_epochs=200, batch_size=8)

