# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:31:54 2021

@author: tav
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.ndimage as sci
from skimage.filters import threshold_otsu, rank, threshold_minimum, difference_of_gaussians, threshold_local, \
    threshold_mean
import skimage.exposure as exposure
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from copy import deepcopy
from PIL import Image
from PIL.TiffTags import TAGS

img = Image.open(os.getcwd() + '/Rec-201220_agamon_flir_100m-349_11_38_44_688_3948.tif')
sh_mes = 3.6
NEAR_FACTOR = 10
NEIGHBORS = 0
NEIGH_DISTANCE = 1 
SHORE_DISTANCE = 2 
HALF = 128

THRESHOLDS = {'201220' : dict(), '030321': dict(), '081121': dict(), '131221': dict(), '231120': dict()}
THRESHOLDS['201220']['thresh_tmp'] = 'f < b[1][0] + 0.8 * (b[1][np.argmax(b[0])] - b[1][0])'
THRESHOLDS['201220']['CRANE_SIZE_THRESH'] = 16
THRESHOLDS['030321']['thresh_tmp'] = 'f < b[1][0] + 0.82 * (b[1][np.argmax(b[0])] - b[1][0])'
THRESHOLDS['030321']['CRANE_SIZE_THRESH'] = 8
THRESHOLDS['081121']['thresh_tmp'] = 'f < b[1][0] + 0.68 * (b[1][np.argmax(b[0])] - b[1][0])'
THRESHOLDS['081121']['CRANE_SIZE_THRESH'] = 7
THRESHOLDS['231120']['thresh_tmp'] = 'f < b[1][0] + 0.66 * (b[1][np.argmax(b[0])] - b[1][0])'
THRESHOLDS['231120']['CRANE_SIZE_THRESH'] = 9

THRESHOLD = None


def analyis(file):
    for thresh in THRESHOLDS: 
        if thresh in file :
            THRESHOLD = THRESHOLDS[thresh]    
    img = Image.open(os.path.join(os.getcwd(),file))
    a = np.average(img)
    e = np.array(img)
    e1 = copy.copy(e)

    plt.imsave('pic_og.bmp', e1, cmap='gray')
    # thresholding
    f = copy.copy(e)
    b = np.histogram(f, np.linspace(np.min(f), np.max(f), 256))
 
    water_tmp = (np.sum(b[0][:]*b[1][+1:]) / np.sum(b[0][:])) #two options for measuring water tmp, average or median, average was better 
    water_tmp_median = b[1][1 + np.argmax(b[0][+1:])] 
    print(water_tmp) 
    print(water_tmp_median)
    # plt.plot(b[1][:-1], b[0]) #plot bi model distributions 
    # plt.pause(1)
    # plt.plot(b[1],(water_tmp_median,0) , "x")
    # plt.text(water_tmp_median*(0.95), 2000, f'water temp median = {water_tmp_median:.2f}', fontsize=10)
    # plt.text(water_tmp-1, 2000, f'water temp avg = {water_tmp:.2f}', fontsize=10)
    # plt.show()

water_tmp = (np.sum(b[0][HALF:]*b[1][HALF+1:]) / np.sum(b[0][HALF:]))

thresh = b[1][0] + 0.8 * (b[1][np.argmax(b[0])] - b[1][0])

bin_img = f < thresh
bin_img = bin_img.astype(np.int32) * 255
plt.imsave('ggg.bmp', bin_img, cmap='gray')
# recontrusting image from 9 parts here

# theshold output overlay on original figure

g = copy.copy(e)

maxg = np.max(g)
ming = np.min(g)

g = 255 * (g - np.min(g)) / (np.max(g) - np.min(g))
g[bin_img == 255] = 255

plt.imsave('ggg2.bmp', g, cmap='gray')

# generate three channel image data, add threshold on top
y = e1[:, :, np.newaxis]
y = np.dstack((y, y, y))
y1 = 255 * (y - np.min(y)) / (np.max(y) - np.min(y))
y1 = y1.astype(np.uint8)

g2 = np.where(bin_img == 0)
plt.imshow(e, cmap='gray')

y1[:, :, 1] = g
plt.imsave('ggg3.bmp', y1, cmap='gray')

# pixl groups labeling

f1 = copy.copy(bin_img)
f1 = bin_img / 255
f1 = f1.astype(np.int32)

s = [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]

labeled_array, num_features = sci.measurements.label(f1, structure=s)
plt.imsave('ggg4.bmp', labeled_array)
plt.imshow(labeled_array)

array_size = np.zeros(num_features + 1)
array_location_mean = np.int32(np.zeros((num_features + 1, 2)))

for i in range(1, num_features + 1):
    array_size[i] = np.count_nonzero(labeled_array == i)
    array_location_mean[i] = np.int32(np.round(np.mean(np.where(labeled_array == i), axis=1)))

array_size_hist = np.histogram(array_size, bins=np.arange(0, 50, 2))

plt.figure()
plt.hist(array_size, bins=np.arange(0, 50, 2))

is_crane = np.ones(num_features + 1)
is_crane[0] = 0
thresh =  16 #array_size_hist[1][np.argmax(array_size_hist[0])] #TODO fix bug not removing small birds 
# find index of land
land_i = np.argmax(array_size)

for i in range(1, num_features + 1):
    #  distance from all points of land
    g4 = np.linalg.norm(array_location_mean[i].reshape((2, 1)) - np.where(labeled_array == land_i), axis=0)


    if np.min(g4) < 10: 
        is_crane[i] = 0
        # ToDo: remove index from array_location_mean and array_size

    # np.argmin(g4) #  closest point in land to array_location_mean[i]

        if array_size[i] < THRESHOLD['CRANE_SIZE_THRESH'] or array_size[i] > THRESHOLD['CRANE_SIZE_THRESH'] * 10:  
            is_crane[i] = 0

crane_location = array_location_mean[np.where(is_crane == 1)]

min_crane_dist = []
crane_neighbor = []

for cr in crane_location:
    crane_dist = np.linalg.norm(crane_location - cr.reshape((1, 2)), axis=1)
    min_temp = np.min(crane_dist[np.where(crane_dist > 0)])  # find closest neighbor
    min_crane_dist.append(min_temp)  
min_temp = np.mean(min_crane_dist)

for cr in crane_location:
    crane_dist = np.linalg.norm(crane_location - cr.reshape((1, 2)), axis=1)
    crane_neighbor.append(np.size(np.where(crane_dist < min_temp * sh_mes)) - 1)  # find how many at sh_mes times the closest

ave_crane_neighbor = np.mean(crane_neighbor)

g7 = copy.copy(y1)

for crane in range(crane_location.shape[0]):
    g7[crane_location[crane][0]][crane_location[crane][1], :] = [255, 0, 0]

plt.imsave('ggg5.bmp', g7, cmap='gray')

#crane = 26
neighbors = {str(crane) : ([],0) for crane in range(len(crane_location))}

g9 = deepcopy(y1)
for crane in range(len(crane_location)) :
    crane_dist = np.linalg.norm(crane_location - crane_location[crane].reshape((1, 2)), axis=1)

    crane_neig_loc = np.where((crane_dist < min_crane_dist[crane] * sh_mes) & (crane_dist > 0))[0]
    crane_neig_dist = crane_location[crane_neig_loc] - crane_location[crane]
    crane_neig_dist_real = list(deepcopy(crane_neig_dist))
    crane_neig_dist_real_tup = [tuple(val) for val in crane_neig_dist_real]

    for i in  range (len(crane_neig_dist)) :
        for j in range (i + 1, len(crane_neig_dist)) :

            a_nor = np.linalg.norm(crane_neig_dist[i])
            b_nor = np.linalg.norm(crane_neig_dist[j])
            ab_nor = np.linalg.norm(crane_neig_dist[i] - crane_neig_dist[j])
            max_cr = crane_neig_dist[i] if (a_nor > b_nor) else crane_neig_dist[j] # farther crane to remove
            deg = (180 / np.pi) * np.arccos(( - (ab_nor ** 2) + (a_nor ** 2) + (b_nor ** 2))
                    / (a_nor * b_nor * 2 ))  #cosine therom
   
            if deg < 30 :  
                if ab_nor < NEAR_FACTOR :  
                    pass
                elif tuple(max_cr) in crane_neig_dist_real_tup:
                    crane_neig_dist_real_tup.remove(tuple(max_cr))

    crane_neig_dist_real = [list(val) for val in crane_neig_dist_real_tup] #python anti magic 
    crane_neig_dist_real +=  crane_location[crane]
    crane_neig_dist_real_tuple = [tuple(val) for val in crane_neig_dist_real]
    crane_location_tuple = [tuple(val) for val in crane_location]
    neighbors[str(crane)] = (crane_neig_dist_real, (np.linalg.norm(crane_neig_dist) / len(crane_neig_dist))) #TODO bug 

    if (crane == 0):
        g99 = deepcopy(y1)
        s_list = [i for i in range (len(crane_location)) if (crane_location_tuple[i] in crane_neig_dist_real_tuple)]

        for crane_neig in s_list: #all cranes neighbor
            g99[crane_location[crane_neig][0]][crane_location[crane_neig][1], :] = [255, 0, 0]

        g99[crane_location[crane][0]][crane_location[crane][1], :] = [0, 0, 0]
        plt.imsave('ggg79.bmp', g99, cmap='gray')

neighbor_distances = [neighbors[str(i)][NEIGH_DISTANCE] for i in range(len(neighbors))]
avg_of_avgs = np.mean(neighbor_distances)

# find cranes closest to shore
print('average distance btween immediate neighbors : ', avg_of_avgs)
close_to_shore = np.ones(len(crane_location), dtype=int)
shore_dist = np.zeros(len(crane_location))

for crane in range(len(crane_location)) :
    g4 = np.linalg.norm(crane_location[crane].reshape((2, 1)) - np.where(labeled_array == land_i), axis=0)

    Shore_point = np.argmin(g4) #  closest point in land to crane_location[crane]
    Shore_point_coor = np.array([np.where(labeled_array == land_i)[0][Shore_point], \
        np.where(labeled_array == land_i)[1][Shore_point]])
 
    crane_shore_dist = Shore_point_coor - crane_location[crane]
    # test if any of the neighbors are closer to the shore
    g99 = deepcopy(y1)
    for neigh in neighbors[str(crane)][NEIGHBORS] :
        ab_nor = np.linalg.norm(neigh - crane_location[crane])
        b_nor = np.linalg.norm(crane_shore_dist)
        a_nor = np.linalg.norm(neigh - crane_location[crane]- crane_shore_dist)
        deg = (180 / np.pi) * np.arccos(( - (a_nor ** 2) + (ab_nor ** 2) + (b_nor ** 2))
                / (ab_nor * b_nor * 2 ))  #cosine theorem

        if deg < 70 : #  has neighbor closer to shore 
            close_to_shore[crane] = 0 
        shore_dist[crane] = b_nor 

shore_hist = []
for i in range (len(crane_location)) : 
    if close_to_shore[i] :
       g99[crane_location[i][0]][crane_location[i][1]] = [0, 0, 0] 
       shore_hist.append(shore_dist[i])
    #g99[Shore_point_coor[0]][Shore_point_coor[1], :] = [255,0,0]
plt.imsave('ggg19.bmp', g99, cmap='gray')


plt.figure(3)
plt.plot(min_crane_dist, crane_neighbor, marker='o', linestyle='None')
plt.show()

# plt.imshow(e)

    print(shore_history)
    print(avg_of_avgs*CM_TO_PXL)
    
    return avg_of_avgs*CM_TO_PXL, std_of_avgs*CM_TO_PXL, shore_history, water_tmp_median

if __name__ == '__main__':
  analyis('Rec-030321_agamon_flir_100m-349_23_26_13_659_7335.tif')