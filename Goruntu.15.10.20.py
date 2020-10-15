#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os

def get_jpeg_files():
    os.getcwd()
    os.listdir()
    path=os.getcwd()
    jpg_files=[f for f in os.listdir(path) if f.endswith('.jpg')]
    return jpg_files

get_jpeg_files()


# In[ ]:


def compare_list_ndarray():
    list_1=[1,"2gugugyufud,3",'4',5,6] #broadcast
    list_2=[2,"2asdasf",'114',15,26]
    print(list_1+list_2)

    list1=[1,2,3]
    list2=[2,3,5]
    print(list1+list2+[10]) #list1+list2+10 error verir

    list3=np.asarray([1,2,3,4])  #ndarray asarray
    list4=np.asarray([1,2,3,4])
    print(list3+list4+10)
    
compare_list_ndarray()


# In[ ]:


def display_two_img(im1,im2):
    plt.subplot(1,2,1)
    plt.imshow(im1)

    plt.subplot(1,2,2)
    plt.imshow(im2)

    plt.show()

def rotate(im1):
    m,n,k=im1.shape
    newimg=np.zeros((n,m,k),dtype='uint8')
    for i in range(m):
        for j in range(n):
            temp1=im1[i,j]
            newimg[j,i]=temp1
    return newimg


# In[ ]:


image2=rotate(image_1)
display_two_img(image_1,image2)


# In[ ]:


t1=10
for i in range(t1):
    for j in range(t1):
        temp1=image_1[i,j]
        print(temp1[2],end=" ") #R G B 


# In[ ]:


image_2=image_1+10
#image_1[10,10,2]+10 #pixel adresleme
image_1.shape,image_2.shape

plt.imshow(image_1)
plt.show()

plt.imshow(image_2)
plt.show()


# In[ ]:




