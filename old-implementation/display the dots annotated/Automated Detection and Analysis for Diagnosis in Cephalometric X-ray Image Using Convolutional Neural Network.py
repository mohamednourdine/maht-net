#!/usr/bin/env python
# coding: utf-8

# ![konya](ktun.png)
# <h4><center>T.C.<br>KONYA TEKNİK ÜNİVERSİTESİ<br>LİSANSÜSTÜ EĞİTİM ENSTİTÜSÜ</center><h4>

# <strong><center>Dental X-ray Image Analysis</center></strong>
# <strong><center>Cephalometric Keypoints Detection using CNN Run</center></strong><br>

# # Automated Detection and Analysis for Diagnosis in Cephalometric X-ray Image Using Convolutional Neural Network

# This project is build on top of the Wang Cwei dataset which can be found in the
# link: https://figshare.com/s/37ec464af8e81ae6ebbf <br>
# 

# ### The scope of this work is to read the images and display the dots annotated by a professional medical doctor

# In[46]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from skimage import io, transform, img_as_float
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, models # add models to the list
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import make_grid
import time
import random

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


# *Lets perfom this operation on one image and sclale before performing a similar operation on multiple images*

# In[91]:


# Take a look at one of the image samples and labels

#NOTE: THE IMAGE FOLDERS HAS BEEN MODIFIED AND SEPERATED INTO TRAIN AND TEST FOLDERS SETS
SAMPLE_PATH = "data/RawImage/Train/TrainingData/005.bmp"
TXT_PATH = "data/AnnotationsByMD/400_senior/005.txt"
# import sample image
img = io.imread(SAMPLE_PATH, as_gray=True)
img


# In[92]:


img.shape


# In[93]:


# import sample coordinates from text as tuples
def extract_labels_from_txt(path):
    with open(path, "r") as f:
        # only first 19 are actual coords in dataset label files
        coords_raw = f.readlines()[:19]
        coords_raw = [tuple([int(float(s)) for s in t.split(",")]) for t in coords_raw]
        return coords_raw


# In[94]:


coords_raw = extract_labels_from_txt(TXT_PATH)
coords_raw


# In[95]:


plt.rcParams["figure.figsize"] = [32,18]
plt.style.use(['dark_background'])
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 1, 1)
ax1.imshow(img, cmap="gray")
# also plot resized image for later 
orig_y, orig_x = img.shape[:2]
SCALE = 15

# for rescale, use same target for both x&y axis
rescaled_img = transform.resize(img,(orig_y/SCALE,orig_y/SCALE))
ax2.imshow(rescaled_img, cmap="gray")

for c in coords_raw:
    # add patches to original image
    # could also just plt.scatter() but less control then
    ax1.add_patch(plt.Circle(c, 5, color='r')) 
    # and rescaled marks to resized images
    x,y = c
    x = int(x*(orig_y*1.0/orig_x)/SCALE)
    y = int(y/SCALE)
    ax2.add_patch(plt.Circle((x,y), 1, color='g')) 

plt.show()


# In[96]:


def print_image(img,labels):
    print(img.shape)
    plt.rcParams["figure.figsize"] = [32,18]
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 1, 1)
    ax1.imshow(img, cmap="gray")
    # also plot resized image for later 
    orig_y, orig_x = img.shape[:2]
    SCALE = 15
    
    # for rescale, use same target for both x&y axis
    rescaled_img = transform.resize(img,(orig_y/SCALE,orig_y/SCALE))
    ax2.imshow(rescaled_img, cmap="gray")
    
    
    for c in coords_raw:
        # add patches to original image
        # could also just plt.scatter() but less control then
        ax1.add_patch(plt.Circle(c, 5, color='r')) 
        # and rescaled marks to resized images
        x,y = c
        x = int(x*(orig_y*1.0/orig_x)/SCALE)
        y = int(y/SCALE)
        ax2.add_patch(plt.Circle((x,y), 1, color='g')) 

    plt.show()


# In[98]:


BASE_IMAGE_PATH='data/RawImage/Train/TrainingData/'
BASE_CORD_PATH='data/AnnotationsByMD/400_senior/'

def display_image_and_cord(image_number,img_path, cord_path):
    data = []
    target = []
    for i, fi in enumerate(os.listdir(img_path)):
           if i<image_number:
                loop_img = io.imread(img_path + fi, as_gray=True)
                lf = fi[:-4] + ".txt"
                loop_labels = extract_labels_from_txt(cord_path + lf)
               
                loop_labels = (np.array(loop_labels))
                print(loop_img)
                print_image(loop_img,loop_labels)
           
        

display_image_and_cord(10,BASE_IMAGE_PATH, BASE_CORD_PATH)


# ## Define transforms
# In the previous section we looked at a variety of transforms available for data augmentation (rotate, flip, etc.) and normalization.<br>
# Here we'll combine the ones we want, including the <a href='https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/22'>recommended normalization parameters</a> for mean and std per channel.

# In[99]:


train_transform = transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# In[100]:


root = 'data/RawImage/'

train_data = datasets.ImageFolder(os.path.join(root, 'Train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'Test'), transform=test_transform)


# In[101]:


train_data


# In[102]:


test_data


# In[103]:


torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)


# In[104]:


class_names = test_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')


# ## Display a batch of images
# To verify that the training loader selects cat and dog images at random, let's show a batch of loaded images.<br>
# Recall that imshow clips pixel values <0, so the resulting display lacks contrast. We'll apply a quick inverse transform to the input tensor so that images show their "true" colors.

# In[105]:


# Grab the first batch of 10 images
for images,labels in train_loader: 
    break


# In[106]:


images


# In[107]:


labels


# In[108]:


# Print the labels
im = make_grid(images, nrow=4)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

# Print the images
plt.figure(figsize=(32,18))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));

