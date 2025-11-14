# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:51:06 2024

@author: JoanaCatarino
"""
import os 
import tifffile
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
atlas_folder = os.path.join(os.environ["USERPROFILE"], '.brainglobe','allen_mouse_10um_v1.2')

#reference_volume = tifffile.imread(os.path.join(atlas_folder, 'reference.tiff'))
annotation_volume = tifffile.imread(os.path.join(atlas_folder, 'annotation.tiff'))

# get volume for only one hemisphere
right_hemis_volume = annotation_volume[:, :, :570].copy()

#print(reference_volume.shape)
print(annotation_volume.shape)
print(right_hemis_volume.shape)

# export right hemisphere volume
tifffile.imwrite('right_hemis_volume.tiff', right_hemis_volume)

#%%
structure_tree = pd.read_csv(os.path.join(atlas_folder, 'structures.csv'))

striatum_rows_by_name = structure_tree[structure_tree.name.str.contains('striatum', case=False)]
print(striatum_rows_by_name)

striatum_rows_by_path = structure_tree[structure_tree.structure_id_path.str.contains('477')]
print(striatum_rows_by_path)

#%% Volume for Caudate Putamen

cpu_volume = annotation_volume == 672
cpu_volume = cpu_volume.astype('uint8')
# export cpu volume
tifffile.imwrite('cpu_volume.tiff', cpu_volume)
# sum 2D plane along AP axis
import numpy as np
cpu_size = np.sum(cpu_volume, axis=(1,2))
print(cpu_size.shape)
print(cpu_size)
import matplotlib.pyplot as plt
plt.plot(cpu_size)

#%% Volume for Nucleus Accumbens 

acb_volume = annotation_volume == 56
acb_volume = acb_volume.astype('uint8')
# export cpu volume
tifffile.imwrite('acb_volume.tiff', acb_volume)
# sum 2D plane along AP axis
import numpy as np
acb_size = np.sum(acb_volume, axis=(1,2))
print(acb_size.shape)
print(acb_size)
import matplotlib.pyplot as plt
plt.plot(acb_size)

#%% Both volumes combined (both hemispheres)

# Define as str volume CP (672) and ACB (56)
str_volume = (annotation_volume == 56) | (annotation_volume == 672)
str_volume = str_volume.astype('uint8')
# export cpu volume
tifffile.imwrite('str_volume.tiff', str_volume)
# sum 2D plane along AP axis
import numpy as np
str_size = np.sum(str_volume, axis=(1,2))
print(str_size.shape)
print(str_size)
import matplotlib.pyplot as plt
plt.plot(str_size)

#%% Both volumes combined (only one hemisphere)

# Define as str volume CP (672) and ACB (56) but only in one hemisphere
str_right_volume = (right_hemis_volume == 56) | (right_hemis_volume == 672)
str_right_volume = str_right_volume.astype('uint8')
# export cpu volume
tifffile.imwrite('str_right_volume.tiff', str_right_volume)
# sum 2D plane along AP axis
import numpy as np
str_size = np.sum(str_right_volume, axis=(1,2))
print(str_size.shape)
print(str_size)
import matplotlib.pyplot as plt
plt.plot(str_size)

#%% How to do the different parcellation?

# from segment 383 (incl.) to segment 561 (incl.) divide in 4 parts
# from segment 562 (incl.) to segment 759 (incl.) divide in 3 parts
# Ignore everuthing befor segment 383 and after segment 759

str_right_volume = tifffile.imread("str_right_volume.tiff")
print(str_right_volume.shape)

empty_hemis_vol = np.zeros((1320, 800, 570), dtype=np.uint8)
print(empty_hemis_vol.shape)

empty_hemis_vol[382:759,:,:] = str_right_volume[382:759,:,:]

hemis_roi = empty_hemis_vol
tifffile.imwrite("hemis_roi.tiff",hemis_roi)

#%%
#now we have volume with clip-out along AP axis, next process first segment from 383 to 561 (179 frames),
# for each frame we divide into 4 parts, we need 4 values top bottom left right, then
# we calculate int((top+bottom)/2) and int((left+right)/2), which gives us horizontal split, vertical split coordinate

# now take frame 383 as an example
frame_383 = hemis_roi[383,:,:]
import matplotlib.pyplot as plt
plt.imshow(frame_383)

# now, find top bottom left right
contours, _ = cv2.findContours(frame_383, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding box for the first contour
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
  
# plot 1    
plt.imshow(frame_383)
plt.axvline(374)    

# plot 2
plt.imshow(frame_383)
plt.axvline(374)
plt.axvline(374+161)
plt.axhline(355)
plt.axhline(355+293)

# plot 3
plt.imshow(frame_383)
plt.axvline(x+0.5*w)
plt.axhline(y+0.5*h)

hsplit, vsplit = int(hsplit), int(vsplit)
frame_383[hsplit:, :vsplit][frame_383[hsplit:, :vsplit] == 1] = 20
frame_383[hsplit:, vsplit:][frame_383[hsplit:, vsplit:] == 1] = 40
frame_383[:hsplit, :vsplit][frame_383[:hsplit, :vsplit] == 1] = 60
frame_383[:hsplit, vsplit:][frame_383[:hsplit, vsplit:] == 1] = 80
plt.imshow(frame_383,cmap="gray")

#%% Do parcellation in 4 for all the 179 frames selected for this

par_4 = np.zeros((179,800,570),dtype=np.uint8)

for frame_idx in range(179):
    save_to = frame_idx
    frame_idx = 382 + frame_idx
    current_frame = hemis_roi[frame_idx,:,:]
    contours, _ = cv2.findContours(current_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    hsplit = int(y + 0.5*h)
    vsplit = int(x + 0.5*w)
    current_frame[hsplit:, :vsplit][current_frame[hsplit:, :vsplit] == 1] = 20
    current_frame[hsplit:, vsplit:][current_frame[hsplit:, vsplit:] == 1] = 40
    current_frame[:hsplit, :vsplit][current_frame[:hsplit, :vsplit] == 1] = 60
    current_frame[:hsplit, vsplit:][current_frame[:hsplit, vsplit:] == 1] = 80
    par_4[save_to,:,:] = current_frame
    print("saved frame", save_to)

tifffile.imwrite("par_4.tiff", par_4)

#%% now generate par_3 with similar idea (198 frames)

frame_562 = hemis_roi[562,:,:]
plt.imshow(frame_562)


contours, _ = cv2.findContours(frame_562, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_, y, _, h = cv2.boundingRect(contours[0])

plt.imshow(frame_562)
plt.axhline(y+h*1/3)
plt.axhline(y+h*2/3)

h1split = int(y+h*1/3)
h2split = int(y+h*2/3)

frame_562[:h1split, :][frame_562[:h1split, :] == 1] = 100
frame_562[h2split:, :][frame_562[h2split:, :] == 1] = 200
frame_562[frame_562 == 1] = 150

plt.imshow(frame_562, cmap="gray")

#%% Do parcellation in 3 for all the 198 frames selected for this 
par_3 = np.zeros((198,800,570),dtype=np.uint8)

for frame_idx in range(198):
    save_to = frame_idx
    frame_idx = 561 + frame_idx
    current_frame = hemis_roi[frame_idx,:,:]
    contours, _ = cv2.findContours(current_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, y, _, h = cv2.boundingRect(contours[0])
    h1split = int(y+h*1/3)
    h2split = int(y+h*2/3)
    current_frame[:h1split, :][current_frame[:h1split, :] == 1] = 100
    current_frame[h2split:, :][current_frame[h2split:, :] == 1] = 200
    current_frame[current_frame == 1] = 150
    par_3[save_to,:,:] = current_frame
    print("saved frame", save_to)

tifffile.imwrite("par_3.tiff", par_3)

#%% Concatenate par4 and par3

# Concatenate along the depth (Z-axis)
final_volume = np.concatenate((par_4, par_3), axis=0)
final_volume = final_volume.astype('uint8')
tifffile.imwrite('final_volume.tiff', final_volume)
str_size = np.sum(final_volume, axis=(1, 2))
print("Final volume shape:", final_volume.shape)  # Should be (par_4 + par_3, 800, 570)
print("Summed size along AP axis:", str_size.shape)
print(str_size)

# Plot the summed values
plt.plot(str_size)
plt.xlabel("Frame index")
plt.ylabel("Summed Intensity")
plt.title("Summed 2D Plane Along AP Axis")
plt.show()

#%%

# Mirror the hemisphere with the par4 and par3 so that we get the parcellation for both hemispheres
final_volume = tifffile.imread('final_volume.tiff')
print(final_volume.shape) # should be (377, 800, 570)
final_volume_mirror = final volume[:,:;::-1] # we want the same AP and DV but the rest of the ML
tifffile.imwrite('final_volume_mirror.tiff', final_volume_mirror)

# Concatenate the 2 hemispheres to get the full Str parcellation
str_2hemis = np.concatenate((final_volume, final_volume_mirror), axis=2) # axis=2 because what we want to concatenate is the ml
print(str_2hemis.shape) # should be (377,800,1140)
tifffile.imwrite('str_2hemis.tiff', str_2hemis)

# Get the final volume with the parcellation by including the str volume in the rest of the brain and set the other regions value to 0
parcel_brain = np.concatenate((np.zeros((382,800,1140)), str_2hemis, np.zeros((561,800,1140))), axis=0)
tifffile.imwrite('parcel_brain.tiff', parcel_brain.astype('uint8'))

#%%

'''
Structures Str

  CP     672   Caudoputamen
  ACB     56   Nucleus accumbens
  DMSt    80   Dorsal Medial Striatum
  DLSt    60   Dorsal Lateral Striatum
  VMSt    40   Ventral Medial Striatum
  VLSt    20   Ventral Lateral Striatum
  DSt    100   Dorsal Striatum
  MSt    150   Medial Striatum
  VSt    200   Ventral Striatum


