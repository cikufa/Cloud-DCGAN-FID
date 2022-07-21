import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image 
from eval import *
import csv
from pathlib import Path 
import os
import imageio
import tensorflow as tf
from utils import *



def preprocess(img_dir, IMAGE_SHAPE, crop_idx=[5,7]): #in: each img/ out: 35(or less) cropped, reshaped, adjusted value images 
  cropped_images= []
  with Image.open(img_dir) as img:
    img = img.resize((crop_idx[0]*IMAGE_SHAPE[0] , crop_idx[1]*IMAGE_SHAPE[1]),
                          Image.ANTIALIAS ) #(1400,2100)->(480, 672)=(5*96,7*96)  
    for c in range(crop_idx[1]): #width , col
      for r in range(crop_idx[0]): #heigh , row 
        cropped = img.crop((r*IMAGE_SHAPE[0], c*IMAGE_SHAPE[1], (r+1)*IMAGE_SHAPE[0], (c+1)*IMAGE_SHAPE[1]))
        cropped_blue=cropped.split()[2]
        if (np.count_nonzero(cropped_blue) == IMAGE_SHAPE[0]*IMAGE_SHAPE[1]):
          #cropped.save(os.path.join("dataset/cropped_images_test", f"cropped{i}_{r}_{c}.jpg"))
          cropped =np.array(cropped.getdata()).reshape(cropped.size[1], cropped.size[0], 3)
          cropped = cropped / 127.5 - 1. #-1<training_data<1 , dtype : float64
          cropped_images.append(cropped) 
  #print("cropped len", len(cropped_images))
  # plt.imshow(cropped_images[0])
  # plt.show()
  return cropped_images


def generate_real_samples(cropped_images, img_num, img_dir, datalist, batch_size):  # returns a batch of fake data(img, label)
  while(len(cropped_images) < batch_size):
    img_num= int(np.random.random() * len(datalist))
    ind= int(np.random.randint(0,15))
    cropped_images = cropped_images + preprocess(datalist[img_num], img_shape)[ind:ind+2]
    
    #img_num = img_num + 1
  real_images = []
  for i in range(batch_size):
    real_images.append(cropped_images.pop())
  real_images = np.array(real_images)
  real_label = np.random.uniform(low= 0.8, high=1.2, size=(batch_size, 1))
  #training_data = np.hstack(training_data, real_label)
  #print("croppeda", len(cropped_images), real_images[0].shape)
  #print("dataloader", len(real_images), real_images[0].shape)
  np.random.shuffle(real_images)
  return cropped_images, img_num, real_images, real_label 
   
  #____________________________________________________________________________________________________________________________________

ebatch = 9
latent_dim = 100
path = 'checkpoints/model'
noise= np.random.normal(0,1,(ebatch, latent_dim))
  
real_ind= [1,4,5,7,10,14,15,16,20,22,23,25,27,28,31,32,34,36,37, 39,40] #21
fake_ind= [2,3,6,8,9,11,12,13,17,18,19,21,24,26,29,30,33,35,38] #19

i=0
img_shape= (128,128,3)
out_pth= 'test/'
u= utils(out_pth, path, img_shape)
data_path= ''
# for g in Path(path).glob('*.h5'):
#   g_model = tf.keras.models.load_model(g)
#   # exs= u.generate_fake_samples(g_model , latent_dim, evalbatch) 
#   exs = (g_model.predict(noise).reshape(ebatch,128,128,3) +1)/2
#   # imageio.imwrite(f'{i}.jpg', exs[0])
#   u.save_plot(exs, out_pth, n=ebatch)
  # i+= 10
#_______________________________________________________________
# g_model = tf.keras.models.load_model('final/model128/generator_080.h5')
# exs = (g_model.predict(noise).reshape(ebatch,128,128,3) +1)/2
# for e in exs:
#   imageio.imwrite(f'test/{i}.jpg', e)
#   i+=1
# g_model = tf.keras.models.load_model('final/model128/generator_060.h5')
# exs = (g_model.predict(noise).reshape(ebatch,128,128,3) +1)/2
# for e in exs:
#   imageio.imwrite(f'test/{i}.jpg', e)
#   i+=1

#batch of real img generate:
# datalist= u.load_data('dataset/train_images','dataset/test_images')
# for i in range(len(real_ind)):
#   cropped_images, img_num, reals, real_label = generate_real_samples([], 0, 'd', datalist, ebatch)
#   u.save_plot(reals, f'test/{real_ind[i]}.jpg')
i=0
for im in os.listdir('../evaluation data/good fake'):
  k= imageio.imread(os.path.join('../evaluation data/good fake', im))
  imageio.imwrite(f'../evaluation data/{fake_ind[i]}', k)