#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

def get_jpeg_files():
    os.getcwd()
    os.listdir()
    path = os.getcwd()
    jpeg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    return jpeg_files
get_jpeg_files()

def display_two_image(im1,im2):
    plt.subplot(1,2,1)
    plt.imshow(im1)
    
    plt.subplot(1,2,2)
    plt.imshow(im2)
    
    plt.show()
    
def rotate_by_i_j_swap(im1):
    m,n,k = im1.shape
    new_img = np.zeros((n,m,k),dtype='uint8')
    for i in range(m):
        for j in range(n):
            temp = im1[i,j]
            new_img[j,i] = temp
    return new_img

image1 = plt.imread('canakkale.jpg')
image2 = rotate_by_i_j_swap(image1)
image3 = rotate_by_i_j_swap(image2)
display_two_image(image1,image2)

def rotate_one_point_with_theta_counterclockwise(point,angle):
    theta = np.radians(angle)
    r = np.array(((np.cos(theta), -np.sin(theta)),
                 (np.sin(theta), np.cos(theta))))
    #print('rotation matrix: ')
    #print(r)
    
    v = np.array(point)
    #print('vector v: ')
    #print(v)
    
    return r.dot(v).astype(int)

def get_all_new_location(im1,angle):
    m,n,k = im1.shape
    new_location_point = []
    for i in range(m):
        for j in range(n):
            new_location_point.append(rotate_one_point_with_theta_counterclockwise([i,j],angle))
    return new_location_point

def get_min_max(new_location_points):
    min_x,min_y = new_location_points[0][0],new_location_points[0][1]
    max_x,max_y = new_location_points[0][0],new_location_points[0][1]
    
    s1 = len(new_location_points)
    for s in range(s1):
        
        if(min_x>new_location_points[s][0]):
            min_x = new_location_points[s][0]
        if(max_x<new_location_points[s][0]):
            max_x = new_location_points[s][0]
            
        if(min_y>new_location_points[s][1]):
            min_y = new_location_points[s][1]
        if(max_y<new_location_points[s][1]):
            max_y = new_location_points[s][1]
    return min_x,min_y,max_x,max_y

new_location_points = get_all_new_location(image1,90)
min_x,min_y,max_x,max_y = get_min_max(new_location_points)
rotate_one_point_with_theta_counterclockwise([0,1],90)


def rotate_an_image(im1,angle):
    m,n,k=im1.shape
    new_location_points = get_all_new_location(im1,angle)
    min_x,min_y,max_x,max_y = get_min_max(new_location_points)
    
    new_m = max_x-min_x+1
    new_n = max_y-min_y+1
    
    x_offset = 0-min_x
    y_offset = 0-min_y
    
    new_image_2 = np.zeros((new_m,new_n,3),dtype='uint8')
    
    for i in range(m):
        for j in range(n):
            new_i,new_j = rotate_one_point_with_theta_counterclockwise([i,j],angle)
            new_image_2[(new_i+(x_offset)),(new_j+(y_offset))] = image1[i,j]
    return new_image_2


i1 = rotate_an_image(image1,116)
plt.imshow(i1)
plt.show()


i1 = rotate_an_image(image1,-20)
plt.imshow(i1)
plt.show()
