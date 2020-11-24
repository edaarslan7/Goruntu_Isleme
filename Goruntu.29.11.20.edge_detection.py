#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def convert_rgb_to_gray(im1): 
    m=im1.shape[0]
    n=im1.shape[1]
    im2=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            im2[i,j]=get_distance(im1[i,j,:])
    return im2
def get_distance(v,w=[1/3,1/3,1/3]):  
    a,b,c=v[0],v[1],v[2]
    w1,w2,w3=w[0],w[1],w[2]
    d=((a**2)*w1 +
      (b**2)*w2 +
      (c**2)*w3)**.5
    return d


# In[ ]:


filepath=u"C:\\Users\\chang\\Downloads\\resim.jpg"
img1=plt.imread(filepath)
img2=convert_rgb_to_gray(img1)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1),plt.imshow(img1)
plt.subplot(1,2,2),plt.imshow(img2,cmap='gray')
plt.show()


# In[ ]:


def get_mask_for_edge():
    return np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3) 

def apply_mask_for_edge(part_if_image):
    mask=get_mask_for_edge()
    return sum(sum(part_if_image*mask))

poi1=get_mask_for_edge()
apply_mask_for_edge(poi1)


# In[ ]:


def get_edges(im_1):
    m=im_1.shape[0]
    n=im_1.shape[1]
    
    im_2=np.zeros((m,n))
    
    for i in range(3,m-3):
        for j in range(3,n-3):
            poi=im_1[i-1:i+2,j-1:j+2]
            im_2[i,j]=apply_mask_for_edge(poi)
    return im_2


# In[ ]:


im_with_edges=get_edges(img2)


# In[ ]:


plt.imshow(im_with_edges,cmap='gray')
plt.show()


# In[ ]:


plt.figure(figsize=(20,30))
plt.subplot(1,3,1),plt.imshow(img1)
plt.subplot(1,3,2),plt.imshow(img2,cmap='gray')
plt.subplot(1,3,3),plt.imshow(im_with_edges,cmap='gray')


# In[ ]:


im_11=plt.imread(r'C:\\Users\\chang\\Downloads\\resim.jpg')
im_12=convert_rgb_to_gray(im_11)
im_with_edges12=get_edges(im_12)


# In[ ]:


plt.figure(figsize=(20,20))
plt.subplot(1,3,1),plt.imshow(im_11)
plt.subplot(1,3,2),plt.imshow(im_12,cmap='gray')
plt.subplot(1,3,3),plt.imshow(im_with_edges12,cmap='gray')


# In[ ]:




