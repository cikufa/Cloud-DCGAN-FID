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
ebatch = 4
latent_dim = 100
path = 'checkpoints/model'
noise= np.random.random((ebatch, latent_dim))
i=0
img_shape= (128,128,3)
out_pth= 'test/'
u= utils(out_pth, path, img_shape)
for g in Path(path).glob('*.h5'):
  g_model = tf.keras.models.load_model(g)
  # exs= u.generate_fake_samples(g_model , latent_dim, evalbatch) 
  exs = (g_model.predict(noise).reshape(ebatch,128,128,3) +1)/2
  # imageio.imwrite(f'{i}.jpg', exs[0])
  u.save_plot(exs, out_pth, n=ebatch)
  # i+= 10
