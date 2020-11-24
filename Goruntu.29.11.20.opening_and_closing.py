#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def convert_RGB_to_monochrome_BW(image1,threshold=100):
    img1=image1
    img2=np.zeros((img1.shape[0],img1.shape[1]))
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if(img1[i,j,0]/3+img1[i,j,1]/3+img1[i,j,1]/3)>threshold:
                img2[i,j]=0
            else:
                img2[i,j]=1
    return img2


# In[ ]:


filepath=u"C:\\Users\\chang\\OneDrive\\Masaüstü\\Adsız1.png"
img1=plt.imread(filepath)

img2=convert_RGB_to_monochrome_BW(img1,0.5)

plt.subplot(1,2,1),plt.imshow(img1)
plt.subplot(1,2,2),plt.imshow(img2,cmap='gray')

plt.show()


# In[ ]:


def define_mask1():
    mask1=[[1,1,1],[1,1,1],[1,1,1]]
    return mask1

def my_dilation(img1,mask,morphologyOperation='dilation'):
    #morphologyOperation can be dilation or erosion
    m=img1.shape[0] 
    n=img1.shape[1] 
    img2=np.zeros((m,n),dtype='uint8')
    for i in range(1,m-1):
        for j in range(1,n-1):
            
            x1=img1[i,j] == mask[1][1]
            
            x2=img1[i-1,j-1] == mask[0][0]
            x3=img1[i-1,j] == mask[0][1]
            x4=img1[i-1,j+1] == mask[0][2]
            
            x5=img1[i+1,j-1] == mask[2][0]
            x6=img1[i+1,j] == mask[2][1]
            x7=img1[i+1,j+1] == mask[2][2]
            
            x8=img1[i,j-1] == mask[1][0]
            x9=img1[i,j+1] == mask[1][2]
            
            if(morphologyOperation=='dilation'):
                r1=x1 or x2 or x3 or x4 or x5
                r2=x6 or x7 or x8 or x9
                
                r=r1 or r2 
            elif(morphologyOperation=='erosion'):
                r1=x1 and x2 and x3 and x4 and x5
                r2=x6 and x7 and x8 and x9
                
                r=r1 and r2 
            img2[i,j]=r
    return img2


# In[ ]:


img3=my_dilation(img2,define_mask1(),'erosion')
img4=my_dilation(img3,define_mask1(),'erosion')
img5=my_dilation(img4,define_mask1(),'erosion')

plt.figure(figsize=(15,15))
plt.subplot(1,3,1),plt.imshow(img1)
plt.subplot(1,3,2),plt.imshow(img2,cmap='gray')
plt.subplot(1,3,3),plt.imshow(img5,cmap='gray')
plt.show()


# In[ ]:


img3=my_dilation(img2,define_mask1(),'dilation')
img4=my_dilation(img3,define_mask1(),'dilation')
img5=my_dilation(img4,define_mask1(),'dilation')

plt.figure(figsize=(15,15))
plt.subplot(1,3,1),plt.imshow(img1)
plt.subplot(1,3,2),plt.imshow(img2,cmap='gray')
plt.subplot(1,3,3),plt.imshow(img5,cmap='gray')
plt.show()


# In[ ]:


img3=my_dilation(img2,define_mask1(),'erosion')  #opening operation
img4=my_dilation(img3,define_mask1(),'erosion')
img5=my_dilation(img4,define_mask1(),'erosion')

img6=my_dilation(img5,define_mask1(),'dilation')
img7=my_dilation(img6,define_mask1(),'dilation')
img8=my_dilation(img7,define_mask1(),'dilation')


plt.figure(figsize=(15,15))
plt.subplot(1,2,1),plt.imshow(img1)
plt.subplot(1,2,2),plt.imshow(img8,cmap='gray')
plt.show()


# In[ ]:


img3=my_dilation(img2,define_mask1(),'dilation')  #closing operation
img4=my_dilation(img3,define_mask1(),'dilation')
img5=my_dilation(img4,define_mask1(),'dilation')

img6=my_dilation(img5,define_mask1(),'erosion')
img7=my_dilation(img6,define_mask1(),'erosion')
img8=my_dilation(img7,define_mask1(),'erosion')


plt.figure(figsize=(15,15))
plt.subplot(1,2,1),plt.imshow(img1)
plt.subplot(1,2,2),plt.imshow(img8,cmap='gray')
plt.show()


# In[ ]:


img9=my_dilation(img8,define_mask1(),'dilation')
img10=my_dilation(img9,define_mask1(),'dilation')
img11=my_dilation(img10,define_mask1(),'dilation')
img12=my_dilation(img11,define_mask1(),'dilation')
img13=my_dilation(img12,define_mask1(),'dilation')




plt.figure(figsize=(15,15))
plt.subplot(1,2,1),plt.imshow(img1)
plt.subplot(1,2,2),plt.imshow(img13,cmap='gray')
plt.show()


# In[ ]:




