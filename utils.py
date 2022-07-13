import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image 
from eval import *
import csv
from pathlib import Path 
class utils():
  def __init__(self, out_pth, model_pth, img_shape):
    self.out_pth= out_pth
    self.model_pth= model_pth
    self.img_shape= img_shape
    self.freq = 10
    self.latent_dim = 100
  
  def load_data(self,pth):
    datalist = []
    for img in tqdm(Path(pth).glob('*.jpg')):
      datalist.append(img)
    return datalist

  # create and save a plot of generated images
  def save_plot(self, examples, path, n=2):
    examples = (examples + 1) / 2.0 #[-1,1] to [0,1]
    for i in range(n * n):
      plt.subplot(n, n, 1 + i)
      plt.axis('off')
      plt.imshow(examples[i])
    #imageio.imwrite(filename)
    plt.savefig(path)
    plt.close()

  # evaluate the discriminator, plot generated images, save generator model
  def summarize_performance(self, epoch,batch, g_model, d_model, datalist, latent_dim, img_dir, fid, batch_size):
    cropped_images=[] 
    img_num = np.random.randint(0,9000) #add test data to dataset
    cropped_images, img_num, X_real, y_real = self.generate_real_samples(cropped_images, img_num, img_dir, datalist, batch_size)
    # evaluate discriminator on real examples
    _ , acc_real = d_model.evaluate(X_real, y_real, verbose=0)
   
    x_fake, y_fake =self.generate_fake_samples(g_model, latent_dim, batch_size)
    # evaluate discriminator on fake examples
    _ , acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance
    #print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real100, acc_fake100))
    self.save_plot(x_fake, path=self.out_pth+'train%03d_%03d.png'%(epoch+1 ,batch+1), n=2)
    
    if epoch%10==0:
        filename = self.model_pth + 'generator%03d_%03d_fid%03d.h5' % (epoch+1, batch+1, fid)
        g_model.save(filename)      

  def preprocess(self, img_dir, IMAGE_SHAPE, crop_idx=[5,7]): #in: each img/ out: 35(or less) cropped, reshaped, adjusted value images 
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


  def generate_real_samples(self, cropped_images, img_num, img_dir, datalist, batch_size):  # returns a batch of fake data(img, label)
    while(len(cropped_images) < batch_size):
      img_num= int(np.random.random() * len(datalist))
      cropped_images = cropped_images + self.preprocess(os.path.join(img_dir, datalist[img_num]), self.img_shape)
      #img_num = img_num + 1
    real_images = []
    for i in range(batch_size):
      real_images.append(cropped_images.pop())
    real_images = np.array(real_images)
    real_label = np.random.uniform(low= 0.8, high=1.2, size=(batch_size, 1))
    #training_data = np.hstack(training_data, real_label)
    #print("croppeda", len(cropped_images), real_images[0].shape)

    #print("dataloader", len(real_images), real_images[0].shape)
    return cropped_images, img_num, real_images, real_label 
   
  #____________________________________________________________________________________________________________________________________

  def generate_fake_samples(self, g_model, latent_dim, batch_size):   # returns a batch of fake data(img, label)
    noise= np.random.normal(0,1,(batch_size,latent_dim))
    # fake_img = g_model.predict(noise)
    fake_img = g_model(noise)
    #y = zeros((n_samples, 1))
    fake_label = np.random.uniform(low= 0.0, high=0.25, size=(batch_size, 1))
    return fake_img, fake_label

  def train(self, img_dir, datalist, g_model, d_model, gan_model, latent_dim, fid,write, n_epochs, batch_size):
    #200 epoch , each epoch 1927 batch , each batch : 128 img (64 real , 64 fake) 
    #bat_per_epo = int(len(datalist) / batch_size)
    bat_per_epo= 300 #30 snydsk code
    half_batch = int(batch_size / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
      print("epoch", i)
      img_num= 0
      cropped_images=[]
      # enumerate batches over the training set
      for j in range(bat_per_epo):
        cropped_images, img_num,  X_real, y_real = self.generate_real_samples(cropped_images, img_num, img_dir, datalist, half_batch)
        d_loss1, _ = d_model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
        
        noise= np.random.normal(0,1,(batch_size,latent_dim))	# predict outputs
        gan_fake_label = np.random.uniform(low= 0.8, high=1.2, size=(batch_size, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(noise, gan_fake_label)
        write.writerow([f'{i}/{j}', d_loss1, d_loss2, g_loss])    
    # evaluate the model performance, sometimes
        # if (j+1) % 100 == 0:
          #print('>ep: %d,  batch: %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
          # summarize_performance(i,j , g_model, d_model, datalist, latent_dim, img_dir)
      #shuffle:  
      np.random.shuffle(datalist)
      
    # calc fid after every f epoch 
      if i % self.freq == 0:
        mu_real1, sigma_real1 = fid.mu_sigma_calc(0, None)
        mu_real2, sigma_real2 = fid.mu_sigma_calc(0, None)
        mu_fake, sigma_fake = fid.mu_sigma_calc(1, g_model)
        fid1 = fid.frechet_distance(mu_real1, mu_fake, sigma_real1, sigma_fake).item()
        fid2 = fid.frechet_distance(mu_real2, mu_fake, sigma_real2, sigma_fake).item()
        fid_mean = (fid1+fid2)/2
        self.summarize_performance(i,j , g_model, d_model, datalist, latent_dim, img_dir, fid_mean , batch_size)
        # print(f"fid r1-r2/ epoch {i}", self.frechet_distance(mu_real1, mu_real2, sigma_real1, sigma_real2).item())
        # print(f"fid r1-f/epoch {i}", self.frechet_distance(mu_real1, mu_fake, sigma_real1, sigma_fake).item())
        # print(f"fid r2-f/epoch {i}", self.frechet_distance(mu_real2, mu_fake, sigma_real2, sigma_fake).item())


