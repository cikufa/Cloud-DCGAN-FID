from unittest.main import MODULE_EXAMPLES
import os
from pathlib import Path
from tqdm import tqdm
import torch
from tqdm.auto import tqdm
import keras
from torch.distributions import MultivariateNormal
from torchvision.models import inception_v3
# !pip install -Uqq ipdb
# import pdb
from torchvision.models import inception_v3
from disc import *
from gen import *
from utils import *

import warnings
warnings.filterwarnings("ignore")

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

model_pth = 'model/'
out_pth = 'out/'
Path(out_pth).mkdir(parents=True, exist_ok=True)
Path(model_pth).mkdir(parents=True, exist_ok=True)
# data_pth = '../Cloud-Segmentation/train_images'
data_pth = 'dataset/train_images'
inception_pth = 'inception_v3_google-1a9a5a14.pth'

datalist = []
for img in tqdm(os.listdir(data_pth)):
  datalist.append(img)
print(len(datalist), datalist[0])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

inception_model = inception_v3(pretrained=False)
inception_model.load_state_dict(torch.load(inception_pth))
inception_model.to(device)
inception_model = inception_model.eval() # Evaluation mode
inception_model.fc = torch.nn.Identity()
# gen.eval()

latent_dim = 100
img_shape=(128,128,3)
utils= utils(out_pth, model_pth, img_shape, inception_model)

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

