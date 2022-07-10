import PIL
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import keras
from torch.distributions import MultivariateNormal
import pandas as pd
import seaborn as sns 
import scipy
from skimage.transform import resize
from torchvision.models import inception_v3
from PIL import Image 
import torchvision.transforms as transforms
import math
from utils import *

class eval():
  def __init__(self, data_pth, datalist, latent_dim, inception_model, util, evalbatch, n_samples):
    self.data_pth = data_pth
    self.datalist = datalist 
    self.latent_dim= latent_dim 
    self.evalbatch= evalbatch
    self.n_samples= n_samples
    self.inception_model= inception_model 
    self.util= util   

  def matrix_sqrt(self, x):
      y = x.cpu().detach().numpy()
      y = scipy.linalg.sqrtm(y)
      return torch.Tensor(y.real, device=x.device)

  def frechet_distance(self, mu_x, mu_y, sigma_x, sigma_y):
      return torch.norm(mu_x-mu_y)+torch.trace(sigma_x + sigma_y - self.matrix_sqrt(sigma_x @ sigma_y)*2)
    
  # def preprocess(img):
  #     img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
  #     return img
      
  def get_covariance(self, features):
      return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

  def plot_fid(self, mu_fake , sigma_fake, mu_real, sigma_real):
    #plot
    indices = [2, 4, 5]
    fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
    fake_samples = fake_dist.sample((20,))
    real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
    real_samples = real_dist.sample((20,))

  #  df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
    df_fake = pd.DataFrame(fake_samples.numpy()) #shape: (smaple, indices)
    # print("fake unique ", df_fake.index.is_unique)
    # print("faek dupl ", df_fake.index.duplicated())
    #df_fake.loc[~df_fake.index.duplicated(), :]
    df_real = pd.DataFrame(real_samples.numpy())
    # print("real  uni ", df_real.index.is_unique)
    # print("real dupl ", df_real.index.duplicated())
    #df_real.loc[~df_real.index.duplicated(), :]

    #df_fake["is_real"] = "no"
    #df_real["is_real"] = "yes"
    #df = pd.concat([df_fake, df_real])
    # plt.plot(df_fake[0])
    # plt.show()
    # plt.plot(df_real[0])
    # plt.show()
    
    # sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')
    # x = np.linspace(mu_fake - 3*sigma_fake, mu_fake + 3*sigma_fake, 100)
    # plt.plot(x, stats.norm.pdf(x, mu_fake, sigma_fake)) 
    # plt.show()
    
      
    df_fake["is_real"] = "no"
    df_real["is_real"] = "yes"
    df = pd.concat([df_fake, df_real])
    sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')
    
  def mu_sigma_calc(self, rf, generator):
    features_list = []
    cur_samples = 0
    imn = int(np.random.random()*(len(self.datalist)-10))
    cropped=[]
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

    with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
    #try:
      #for real_example, _ in tqdm(dataloader, total=n_samples // batch_size): # Go by batch
      #for real_samples in tqdm(dataloader):
      for i in range(0, self.n_samples//self.evalbatch, self.evalbatch):
      # while cur_samples < n_samples:
        if (rf == 0): #getin real image features  
          cropped, imn, real_samples, reallbl  = self.util.generate_real_samples(cropped_images= cropped,
              img_num=imn, img_dir=self.data_pth, datalist=self.datalist, batch_size = self.evalbatch) #real sample: (evalbatch, 128*128 ,3)
          real_samples= (real_samples + 1)//2  # [-1,1] -> [0,1]
          real_samples= real_samples.reshape((self.evalbatch,128,128,3)) 
          real_samples= resize(real_samples, (self.evalbatch, 299, 299))
          real_samples= real_samples.reshape((self.evalbatch,3,299,299))
          real_samples= torch.tensor(real_samples).float()
          #real_samples = torch.nn.functional.interpolate(real_samples, size=(299, 299), mode='bilinear', align_corners=False) #RECENTLY ADDED
          inception_model = self.inception_model.float()
          real_features = self.inception_model(real_samples.to(device)).detach().to('cpu') # (evalbatch,2048)
          features_list.append(real_features)
          
        elif rf == 1: #getin fake image features  
          noise = np.random.random((self.evalbatch, self.latent_dim))
          fake_samples = (generator(noise) + 1)/2  #(evalbatch, 128,128,3)
          #fake_samples= torch.tensor(fake_samples).reshape(evalbatch,3,128,128)
          #fake_samples = torch.nn.functional.interpolate(fake_samples, size=(299, 299), mode='bilinear', align_corners=False)
          fake_samples= resize(fake_samples, (self.evalbatch, 299, 299)) #(evalbatch, 299,299,3)
          fake_samples= fake_samples.reshape((self.evalbatch,3,299,299))
          fake_samples= torch.tensor(fake_samples)
          fake_features = self.inception_model(fake_samples.to(device)).detach().to('cpu')
          features_list.append(fake_features)  #(n_samples, 2048)

        #cur_samples += evalbatch      
    #except:
    # else:
    #   print("Error in loop")
    
    features_all = torch.cat(features_list)
    mu = torch.mean(features_all, 0) #(1,2048)
    sigma = self.get_covariance(features_all) #(2048,2048)
    # print("mu , sigma", mu.shape, sigma.shape)
    # print("features", len(features_list), features_list[0].shape)
    return mu, sigma
