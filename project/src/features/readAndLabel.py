import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Note: I stored all .nii files in the directory 'project\volume\data\raw'
# but I use gitignore to avoid updating everything in volume folder (including those 262 .nii files)
# except for the output of this file


# change the current working directory from 'project\src\features' to 'project\volume\data\raw'
os.chdir(os.path.abspath('..'))
os.chdir(os.path.abspath('..\\volume\\data\\raw'))

# contains directory for all .nii files in 'project\volume\data\raw'
originalFiles = [os.getcwd() + '\\' + i for i in os.listdir(os.getcwd())
                # the following line is referenced from Liam's Branch, which guarantees no files other than .nii files will be read
                 if i.endswith(".nii") and os.path.isfile(os.getcwd() + '\\' + i)
                 ]

# change the current working directory from 'project\volume\data\raw' to 'project\src\features'
os.chdir(os.path.abspath('..'))
os.chdir(os.path.abspath('..'))
os.chdir(os.path.abspath('..\\src\\features'))

# a list stores directory for all files in form of segmentation-n.nii in 'project\volume\data\raw', for all n from 0 to 130
segmentationList = []
# a list stores directory for all files in form of volume-n.nii in 'project\volume\data\raw', for all n from 0 to 130
volumeList = []

for i in originalFiles:
    if re.search('segmentation', i[-20:]):
        segmentationList.append(i)
    elif re.search('volume', i[-20:]):
        volumeList.append(i)


# a dictionary stores directory for all files in 'project\volume\data\raw'
fileDictionary = {}

# structure:
# fileDictionary = {
#       n: [segmentation-n.nii, volume-n.nii]
# }
# where n is integer from 0 to 130 since len(segmentationList) = len(volumeList) = 131
# and segmentation-n.nii and volume-n.nii represents the absolute directory of the specific file

# check if we have same amount of segmentation files and volume files
if len(segmentationList) == len(volumeList):
    print('creating fileDictionary...')
    for i in range(len(segmentationList)):
        # if i not in fileDictionary.keys():
        #     fileDictionary[i] = []
        tempPattern = '-' + str(i) + '.'
        for j in segmentationList:
            if re.search(tempPattern, j[-8:]):
                fileDictionary[i] = [j]
        for k in volumeList:
            if re.search(tempPattern, k[-8:]):
                fileDictionary[i].append(k)
    print('fileDictionary created!')
# print error message
else:
    print('The amount of volumes and segmentations does not match.')

# for i in fileDictionary:
#     print(i, fileDictionary[i])


# a dictionary stores the combination of segmentation and volume data
labeledDictionary = {}

# structure:
# labeledDictionary
# {
#       n: [
#              [
#                  [
#                      [corresponding value in segmentation.nii, corresponding value in volume-n.nii]
#                  ]
#              ]
#          ]
# }
# where n is integer from 0 to 130 since len(segmentationList) = len(volumeList) = 131


## Note: the current structure/method is very slow due to the huge size of each numpy.memmap


#for i in range(len(segmentationList)):
# for testing purpose, only run following codes for one patient (patient with number 99)
for i in range(99, 100):
    segImg = nib.load(fileDictionary[i][0])
    volImg = nib.load(fileDictionary[i][1])
    segData = segImg.get_fdata()
    volData = volImg.get_fdata()

    labeledDictionary[i] = []

    # comment out line 102 - 113 to avoid wasting of time
    # you can print segData/volData or their shape to see their structure
    for l in range(len(segData)):
        labeledDictionary[i].append([])
        for m in range(len(segData[l])):
            labeledDictionary[i][l].append([])
            for n in range(len(segData[l][m])):
                labeledDictionary[i][l][m].append([])
                labeledDictionary[i][l][m][n] = [segData[l][m][n], volData[l][m][n]]
    print(labeledDictionary[i][1][1][1])
    #print('shapes:', labeledDictionary[i].shape)
    print('data')
    print(segData[1][1][1])
    print(volData[1][1][1])

