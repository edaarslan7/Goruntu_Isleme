#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.getcwd()
os.listdir()


# In[ ]:


path=os.getcwd()
jpg_files=[f for f in os.listdir(path) if f.endswith('.jpg')]
jpg_files


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def get_value_from_triple(temp1):
    #temp1=im1[0,0,:]
    return int(temp1[0]/3+temp1[1]/3+temp1[2]/3)
def get_0_1_from_triple(temp1):
    #temp1=im1[0,0,:]
    temp=int(temp1[0]/3+temp1[1]/3+temp1[2]/3)
    if(temp<110):
        return 0
    else:
        return 1


# In[ ]:


def convert_rgb_to_bw(im1):
    m,n,k=im1.shape
    newimg=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_0_1_from_triple(im1[i,j,:])
            newimg[i,j]=s
    return newimg
def convert_rgb_to_gray(im1):
    m,n,k=im1.shape
    newimg=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_value_from_triple(im1[i,j,:])
            newimg[i,j]=s
    return newimg


# In[ ]:


#plt.imsave('171112220-352-k290611.jpg',im2,cmap='gray')
#plt.imsave('171112220-352-k290611.jpg',im1_bw,cmap='gray')
im1=plt.imread(jpg_files[0])
print(im1.shape)
im1_gray=convert_rgb_to_gray(im1)
im1_bw=convert_rgb_to_bw(im1)


# In[ ]:


plt.subplot(1,3,1)
plt.imshow(im1)

plt.subplot(1,3,2)
plt.imshow(im1_gray,cmap='gray')

plt.subplot(1,3,3)
plt.imshow(im1_bw,cmap='gray')

plt.show()


# In[ ]:




