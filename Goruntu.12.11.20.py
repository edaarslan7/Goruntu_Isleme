#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.getcwd()


# In[ ]:


path=r"C:\Users\chang\Downloads"
files_path=path+"\\resim.jpg"
files_path


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


img_0=plt.imread(files_path)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img_0)
plt.show()


# In[ ]:


#max(img_0) #boyut fazla olduğunda hata verir
np.max(img_0)


# In[ ]:


img_0.ndim,img_0.shape


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


def myf(a,b):
    assert a>=0;" intensity positive", "error intensity not positive"
    if(a<=255-b):
        return a+b
    else:
        return 255
myf(3,123)


# In[ ]:


img_1=convert_rgb_to_gray(img_0)
plt.imshow(img_1,cmap='gray')
plt.show()


# In[ ]:


img_1.ndim,img_1.shape


# In[ ]:


m,n = img_1.shape
img_2 = np.zeros((m,n),dtype="uint8")


# In[ ]:


def myf2(a):
    return int(255-a)
myf2(255)


# In[ ]:


for i in range(m):
    for j in range(n):
        intensity=img_1[i,j]
        #increment=50
        #print(intensity)
        #img_2[i,j]=myf(intensity,increment)
        img_2[i,j]=myf2(intensity)


# In[ ]:


plt.subplot(2,2,1),plt.imshow(img_0,cmap='gray')
plt.subplot(2,2,2),plt.imshow(img_1,cmap='gray')
plt.subplot(2,2,3),plt.imshow(img_2,cmap='gray')
plt.show()


# In[ ]:


np.min(img_2),np.max(img_2)


# In[ ]:


x=np.array(list(range(100)))
#y=np.array(list(range(100)))
#y=np.sin(np.array(list(range(100))))
#y=1/(1+np.exp(x))
y1=np.power(x/float(np.max(x)),1)
y2=np.power(x/float(np.max(x)),10)
y3=np.power(x/float(np.max(x)),1/10)


plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)


# In[ ]:


def myf3(img100,gamma):
    return np.power(img100/float(np.max(img100)),gamma)

img_100=myf3(img_0,10)

plt.imshow(img_100)
plt.show()


# In[ ]:




