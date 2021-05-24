#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:11:59 2021

@author: dgatti
"""
# ### Calculate lung masks and Covid vs non-Covid scores 
# 
# To launch from the command line:  
# /Path to python executable inside virtual env/python g_CXR_Net.py --xdir 'directory where this script resides'

# In[0]:
import os, sys, shutil, getopt
from os import listdir, path
from os.path import isfile, join
import argparse
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import pydicom
from pydicom.data import get_testdata_files
import random
import numpy as np
import cv2
from threading import Thread
import json
from json import dump, load
import datetime
import csv, h5py
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json, load_model, clone_model 
from tensorflow.keras.layers import Input, Average, Lambda, Multiply, Add, GlobalAveragePooling2D, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image

# In[1]:
    
# Initialize the Parser
parser = argparse.ArgumentParser(description ='Executable location') 
   
parser.add_argument('--xdir', metavar = 'D', dest = 'xdir',
                    type = str, nargs = 1,
                    help ='directory of the gui executable')

args = parser.parse_args()    
    
launchdir = args.xdir[0]
currentdir = os.getcwd()
if currentdir != launchdir:
    os.chdir(launchdir)

# In[2]:
main_window = Tk()
main_window.title("CXR-Net")
main_window.geometry('1260x700') # 1260x680
main_window.resizable(width=True, height=True)
# main_window.configure(bg='#E5E5E5')

# Labels
title_0_0 = Label(main_window,text='DIRECTORIES (if none entered defaults are used)',
               font=('Helvetica',12,'bold'),fg='blue')
title_0_0.grid(row=0,column=0,columnspan=2)

title_0_4 = Label(main_window,text='IMAGES (if none entered all images are used)',
               font=('Helvetica',12,'bold'),fg='blue')
title_0_4.grid(row=0,column=3,columnspan=2)


def selectdir():
    filename = filedialog.askdirectory(initialdir=os.getcwd(), parent=None )
    return filename

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename  

def openfns():
    filename = filedialog.askopenfilenames(title='open')
    return filename


entries = ['','','','','','','','','']

# BUTTON_1_0
dirlist1 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist1.grid(row=1,column=1,rowspan=1,columnspan=1,pady=10,padx=5) 
def open_dir1():
    global entries
    entries[0] = selectdir().split('/')[-1]   
    dirlist1.delete(0,END)
    dirlist1.insert(END,entries[0])
        
button_1_0 = Button(main_window, text='Root', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir1)
button_1_0.grid(row=1,column=0,padx=10)

# BUTTON_2_0
dirlist2 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist2.grid(row=2,column=1,rowspan=1,columnspan=1,pady=10) 
def open_dir2():
    global entries
    entries[1] = selectdir().split('/')[-1]
    dirlist2.delete(0,END)
    dirlist2.insert(END,entries[1])    
    
button_2_0 = Button(main_window, text='DCM images', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir2)
button_2_0.grid(row=2,column=0)

# BUTTON_3_0
dirlist3 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist3.grid(row=3,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir3():
    global entries
    entries[2] = selectdir().split('/')[-1]
    dirlist2.delete(0,END)
    dirlist2.insert(END,entries[1])    
    
button_3_0 = Button(main_window, text='PNG images', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir3)
button_3_0.grid(row=3,column=0)

# BUTTON_4_0
dirlist4 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist4.grid(row=4,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir4():
    global entries
    entries[3] = selectdir().split('/')[-1]
    dirlist4.delete(0,END)
    dirlist4.insert(END,entries[3])
    
button_4_0 = Button(main_window, text='Bin. masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir4)
button_4_0.grid(row=4,column=0)

# BUTTON_5_0
dirlist5 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist5.grid(row=5,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir5():
    global entries
    entries[4] = selectdir().split('/')[-1]
    dirlist5.delete(0,END)
    dirlist5.insert(END,entries[4])    
    
button_5_0 = Button(main_window, text='Flp. masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir5)
button_5_0.grid(row=5,column=0)


# BUTTON_6_0
dirlist6 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist6.grid(row=6,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir6():
    global entries
    entries[5] = selectdir().split('/')[-1]
    dirlist6.delete(0,END)
    dirlist6.insert(END,entries[5])    
    
button_6_0 = Button(main_window, text='Imgs/masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir6)
button_6_0.grid(row=6,column=0)

# BUTTON_7_0
dirlist7 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist7.grid(row=7,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir7():
    global entries
    entries[6] = selectdir().split('/')[-1]
    dirlist7.delete(0,END)
    dirlist7.insert(END,entries[6])    
    
button_7_0 = Button(main_window, text='H5', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir7)
button_7_0.grid(row=7,column=0)

# BUTTON_8_0
dirlist8 = Listbox(main_window,height=1,width=25,borderwidth=0.5)
dirlist8.grid(row=8,column=1,rowspan=1,columnspan=1,pady=10)
def open_dir8():
    global entries
    entries[7] = selectdir().split('/')[-1]
    dirlist8.delete(0,END)
    dirlist8.insert(END,entries[7])      
    
button_8_0 = Button(main_window, text='Heat maps', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir8)
button_8_0.grid(row=8,column=0)


# BUTTON_1_3
filelist1 = Listbox(main_window,height=1,width=90,borderwidth=0.5)
filelist1.grid(row=1,column=4,rowspan=1,columnspan=2,pady=10,padx=5)   
def open_file9():
    global entries
    entries[8] = openfns()
    global selected_imgs    
    selected_imgs = []
    for selected_name in entries[8]:
        selected_imgs.append(selected_name.split('/')[-1])

    print(entries[8])
    print(selected_imgs)          
  
    filelist1.delete(0,END)
    filelist1.insert(END,selected_imgs) 
        
button_1_3 = Button(main_window, text='DCM names', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_file9)
button_1_3.grid(row=1,column=3,padx=10)

 
# BUTTON_9_0
dirlist = Listbox(main_window,height=16,width=25,borderwidth=0.5)
dirlist.grid(row=9,column=1,rowspan=4,columnspan=2,pady=20)
vscrollbar = Scrollbar(main_window,orient=VERTICAL)
vscrollbar.grid(row=9,column=3,rowspan=3)
dirlist.configure(yscrollcommand=vscrollbar.set)
vscrollbar.configure(command=dirlist.yview)
hscrollbar = Scrollbar(main_window,orient=HORIZONTAL)
hscrollbar.grid(row=13,column=1,columnspan=2)
dirlist.configure(xscrollcommand=hscrollbar.set)
hscrollbar.configure(command=dirlist.xview)

def on_click(): 
           
    # Generate all the directories paths
    global entries, root_dir, dcm_source_img_path, source_resized_img_path, \
        target_resized_msk_path_binary, target_resized_msk_path_float,\
        target_img_mask_path, h5_img_dir, H5_IMAGE_DIR, grad_cam_dir, valid_image_name,\
        new_patient_h5_dict, new_patient_dcm_name_list, selected_imgs
    
    if entries[0]=='': 
        entries[0]='new_patient_cxr'
        dirlist1.insert(END,entries[0])
    if entries[1]=='': 
        entries[1]='image_dcm'
        dirlist2.insert(END,entries[1])
    if entries[2]=='': 
        entries[2]='image_resized_equalized'
        dirlist3.insert(END,entries[2])        
    if entries[3]=='': 
        entries[3]='mask_binary'
        dirlist4.insert(END,entries[3])        
    if entries[4]=='': 
        entries[4]='mask_float'
        dirlist5.insert(END,entries[4])        
    if entries[5]=='': 
        entries[5]='image_mask'
        dirlist6.insert(END,entries[5])        
    if entries[6]=='': 
        entries[6]='H5'
        dirlist7.insert(END,entries[6])        
    if entries[7]=='': 
        entries[7]='grad_cam'
        dirlist8.insert(END,entries[7]) 
        
    
    # ### SOURCE and TARGET DIRECTORIES.
    
    # Root directory
    root_dir = entries[0] # 'new_patient_cxr/
    
    # Source directory containing COVID patients lung CXR's
    dcm_source_img_path = os.path.join(root_dir, entries[1]) # 'new_patient_cxr/image_dcm/'
    
    # Source/target directory containing COVID patients lung DCM CXR's converted to PNG
    source_resized_img_path = os.path.join(root_dir, entries[2]) # 'new_patient_cxr/image_resized_equalized_from_dcm/'
    
    # Target directories for predicted masks
    target_resized_msk_path_binary = os.path.join(root_dir, entries[3]) # 'new_patient_cxr/mask_binary/'
    target_resized_msk_path_float = os.path.join(root_dir, entries[4]) # 'new_patient_cxr/mask_float/'
    target_img_mask_path = os.path.join(root_dir, entries[5]) # 'new_patient_cxr/image_mask/'
    
    # Target directory for H5 file containing both images and masks
    H5_IMAGE_DIR = os.path.join(root_dir, entries[6])
    
    # Target directory for gradcams
    grad_cam_dir = os.path.join(root_dir, entries[7]) # 'new_patient_cxr/grad_cam/'
    
    
    # Read in the patient list and generate json dictionary for the patient names   
    source_img_names = [f for f in listdir(dcm_source_img_path) if isfile(join(dcm_source_img_path, f))]    
    print(source_img_names)
    
    new_patient_dcm_name_list = []
    
    for valid_image_name in source_img_names:
        if valid_image_name == '.DS_Store': 
            continue
        if valid_image_name == '__init__.py': 
            continue 
        if (entries[8]!='') and (valid_image_name not in selected_imgs):
            continue    

        valid_image_name = valid_image_name[:-3] + 'h5'
        new_patient_dcm_name_list.append(valid_image_name) 
    
    new_patient_h5_dict = {"new_patient":new_patient_dcm_name_list}
    
    # data set
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'), 'w') as filehandle:
        json.dump(new_patient_h5_dict, filehandle)     
    
    # Read in 'new patient list'
    H5_IMAGE_DIR = os.path.join(entries[0], entries[6])
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle:
        new_patient_h5_dict = json.load(filehandle)      

    dirlist.delete(0,END)
    for row in entries[:8]:
        dirlist.insert(END,row)  
        
    for row in new_patient_h5_dict["new_patient"]:
        row = row[:-2] + 'dcm'
        dirlist.insert(END,row)        
        
button_9_0 = Button(main_window,text='Update',font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=10,
        command=on_click)
button_9_0.grid(row=9,column=0,pady=10)


# BUTTON_10_0
def get_masks_threaded():
    Thread(target=get_masks).start()

def get_masks():

    global entries, root_dir, dcm_source_img_path, source_resized_img_path, \
        target_resized_msk_path_binary, target_resized_msk_path_float,\
        target_img_mask_path, h5_img_dir, H5_IMAGE_DIR, grad_cam_dir, valid_image_name,\
        new_patient_h5_dict, new_patient_dcm_name_list, model      
    
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle:
        new_patient_h5_dict = json.load(filehandle)     

    # dirlist.delete(0,END)
    dirlist.insert(END,'')    
    dirlist.insert(END,'Calculating masks')  

###--------------------Beginning of GET_MASKS subroutine-----------------------    

# In[1m]:
      
    # from MODULES_1.Generators import train_generator_1, val_generator_1
    # from MODULES_1.Generators import train_generator_2, val_generator_2
    from MODULES_1.Networks import ResNet_Atrous
    from MODULES_1.Losses import dice_coeff
    from MODULES_1.Losses import tani_loss, tani_coeff, weighted_tani_coeff
    from MODULES_1.Losses import weighted_tani_loss
    from MODULES_1.Constants import _Params, _Paths
    # from MODULES_1.Utils import get_class_threshold
    # from MODULES_1.Utils import get_model_memory_usage
    
# In[2m]:
    
    # ### CONSTANTS
    
    HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,TRAIN_SIZE,VAL_SIZE,TEST_SIZE,DR1,DR2,CLASSES,IMG_CLASS = _Params()
    
    TRAIN_IMG_PATH,TRAIN_MSK_PATH,TRAIN_MSK_CLASS,VAL_IMG_PATH,VAL_MSK_PATH,VAL_MSK_CLASS,TEST_IMG_PATH,TEST_MSK_PATH,TEST_MSK_CLASS = _Paths()
    
# In[3m]: 
    
    # ### LOAD LUNG SEGMENTATION MODEL AND COMPILE
    if ('model' in locals()) or ('model' in globals()):
        print('Segmentation model already loaded')
    else:
        print('Loading and compiling segmentation model')
        
        model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'
        model_number = '2020-10-16_21_26' # model number from an earlier run
        filepath = 'models/' + model_selection + '_' + model_number + '_all' + '.h5'
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = load_model(filepath, compile=False)     
            model.compile(optimizer=Adam(), loss=weighted_tani_loss, metrics=[tani_coeff]) 
    
        print(f'Model selection: {model_selection}')
        print(f'Model number: {model_number}')
    
# In[4m]:
    
    # get CXR DCM image names from source directory and convert to png
    
    print(f'DCM source: {dcm_source_img_path}')
    print(f'PNG source: {source_resized_img_path}')
    
    source_img_names = [f for f in listdir(dcm_source_img_path) if isfile(join(dcm_source_img_path, f))]

    for name in source_img_names:
        # print(f'Image name: {name}')
        if name == '.DS_Store': 
            continue
        if name == '__init__.py': 
            continue    
        if (entries[8]!='') and (name not in selected_imgs):
            continue     
               
        print(f'DCM image: {name}')   
        filename = os.path.join(dcm_source_img_path, name)
        dataset = pydicom.dcmread(filename)    
                            
        # Write PNG image    
        img = dataset.pixel_array.astype(float)    
        minval = np.min(img)
        maxval = np.max(img)
        scaled_img = (img - minval)/(maxval-minval) * 255.0
        
        WIDTH = 340
        HEIGHT = 300
    
        resized_img = cv2.resize(scaled_img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
        resized_img_8bit = cv2.convertScaleAbs(resized_img, alpha=1.0)
        equalized_img = cv2.equalizeHist(resized_img_8bit)    
        

        new_name = name.replace('.dcm', '.png')
            
        cv2.imwrite(os.path.join(source_resized_img_path, new_name), equalized_img) 
        
        print(f'PNG image: {new_name}')
    
        expanded_img = equalized_img/255    
        expanded_img = np.expand_dims(expanded_img,axis = [0,-1])
        
        print(f'Expanded img dims = {expanded_img.shape}')

        # Run the segmentation model
        mask = model(expanded_img).numpy()
        
        mask_float = np.squeeze(mask[0,:,:,0])    
        mask_binary = (mask_float > 0.5)*1
                  
        mask_float *=255    
        mask_binary *=255
        cv2.imwrite(os.path.join(target_resized_msk_path_float, new_name), mask_float)
        cv2.imwrite(os.path.join(target_resized_msk_path_binary, new_name), mask_binary)
    
        mask_float2 = cv2.imread(os.path.join(target_resized_msk_path_float, new_name), cv2.IMREAD_GRAYSCALE)    
        mask_binary2 = cv2.imread(os.path.join(target_resized_msk_path_binary, new_name), cv2.IMREAD_GRAYSCALE)      
        img_mask = cv2.hconcat([equalized_img,mask_float2,mask_binary2])
        cv2.imwrite(os.path.join(target_img_mask_path, new_name[:-4] + '_img_and_pred_mask.png'), img_mask)    
            
# In[5m]:
    
    # ### RECOVER STANDARDIZATION PARAMETERS FROM MODULE 1
    
    with open('standardization_parameters_V7.json') as json_file:
        standardization_parameters = json.load(json_file) 
        
    train_image_mean = standardization_parameters['mean']
    train_image_std = standardization_parameters['std']
    
# In[6m]:
    
    # ### PREPARE STANDARDIZED IMAGES as H5 FILES
    
    # Source directories for images and masks
    IMAGE_DIR = source_resized_img_path
    MASK_DIR = target_resized_msk_path_float    
        
    # Loop over the set of images to predict
    
    # get CXR image names from source directory
    source_img_names = [f for f in listdir(source_resized_img_path) if isfile(join(source_resized_img_path, f))]
    
    for valid_image_name in source_img_names:
        # print(f'Image name: {name}')
        if valid_image_name == '.DS_Store': 
            continue
        if valid_image_name == '__init__.py': 
            continue    
        if (entries[8]!='') and (valid_image_name[:-4] + '.dcm' not in selected_imgs):
            continue    
        
        # Radiologist labels and class weights are not known
        valid_pos_label = 0
        valid_neg_label = 0
        valid_weight = 1.0
    
        valid_image = cv2.imread(os.path.join(IMAGE_DIR, valid_image_name), cv2.IMREAD_GRAYSCALE)            
        valid_image = np.expand_dims(valid_image,axis=-1)
            
        # External learned mask of segmented lungs
        valid_learned_mask = cv2.imread(os.path.join(MASK_DIR, valid_image_name), cv2.IMREAD_GRAYSCALE).astype('float64')
        valid_learned_mask /= 255
        valid_learned_mask = np.expand_dims(valid_learned_mask,axis=-1)
        
        # Internal thresholded mask    
        low_ind = valid_image < 6
        high_ind = valid_image > 225    
        valid_thresholded_mask = np.ones_like(valid_image)
        valid_thresholded_mask[low_ind] = 0
        valid_thresholded_mask[high_ind] = 0
    
        # Combine the two masks
        valid_mask = np.multiply(valid_thresholded_mask,valid_learned_mask)
        
        # Standardization with training mean and std 
        valid_image = valid_image.astype(np.float64)
        valid_image -= train_image_mean
        valid_image /= train_image_std        
        
        with h5py.File(os.path.join(H5_IMAGE_DIR, valid_image_name[:-4] + '.h5'), 'w') as hf: 
            # Images
            Xset = hf.create_dataset(
                name='X',
                data=valid_image,
                shape=(HEIGHT, WIDTH, 1),
                maxshape=(HEIGHT, WIDTH, 1),
                compression="gzip",
                compression_opts=9)
            
            # Masks
            Mset = hf.create_dataset(
                name='M',
                data=valid_mask,
                shape=(HEIGHT, WIDTH, 1),
                maxshape=(HEIGHT, WIDTH, 1),
                compression="gzip",
                compression_opts=9)
            
            # Labels
            yset = hf.create_dataset(
                name='y',
                data=[valid_pos_label,valid_neg_label])
            
            # Class weights
            wset = hf.create_dataset(
                name='w',
                data=valid_weight)             
    
# In[7m]:
    
    # ### Generate json dictionary for the new patient names
    
    new_patient_h5_name_list = []
    
    for valid_image_name in source_img_names:
        if valid_image_name == '.DS_Store': 
            continue
        if valid_image_name == '__init__.py': 
            continue    
        if (entries[8]!='') and (valid_image_name[:-4] + '.dcm' not in selected_imgs):
            continue    
    
        new_patient_h5_name_list.append(valid_image_name[:-4] + '.h5') 
    
    new_patient_h5_dict = {"new_patient":new_patient_h5_name_list}
    
    # data set
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'), 'w') as filehandle:
        json.dump(new_patient_h5_dict, filehandle)     
    
    print(f'H5 dataset: {new_patient_h5_dict["new_patient"]}')
    
###--------------------------end of GET_MASKS subroutine-----------------------

    # dirlist.insert(END,'')
    dirlist.insert(END,'Mask calculation completed')

button_10_0 = Button(main_window,text='Get masks',font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=10,
       command=get_masks)
button_10_0.grid(row=10,column=0)


# BUTTON_11_0
def get_scores_threaded():
    Thread(target=get_scores).start()

def get_scores():
    
    # dirlist.delete(0,END)
    dirlist.insert(END,'')    
    dirlist.insert(END,'Calculating scores')    

###---------------------Beginning of Get_scores subroutine--------------------    
   
# In[1s]:
    
    global entries, root_dir, dcm_source_img_path, source_resized_img_path, \
        target_resized_msk_path_binary, target_resized_msk_path_float,\
        target_img_mask_path, h5_img_dir, H5_IMAGE_DIR, grad_cam_dir, valid_image_name,\
        new_patient_h5_dict, new_patient_dcm_name_list, ensemble_model,\
        valid_1_generator, new_patient_list
        
# In[2s]:
    
    # PREDICTION and HEAT MAP
    
    from MODULES_2.Generators import get_generator, DataGenerator
    from MODULES_2.Networks import WaveletScatteringTransform, ResNet 
    from MODULES_2.Networks import SelectChannel, TransposeChannel, ScaleByInput, Threshold 
    # from MODULES_2.Losses import other_metrics_binary_class
    from MODULES_2.Constants import _Params, _Paths
    # from MODULES_2.Utils import get_class_threshold, standardize, commonelem_set
    # from MODULES_2.Utils import _HEAT_MAP_DIFF   
        
# In[]:
        
    # ### Read in json dictionary for the new patient names
    # with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle:
    #     new_patient_h5_dict = json.load(filehandle)     
    
# In[3s]:
    
    # ### MODEL AND RUN SELECTION
    HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,\
    DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,SHIFT_LIMIT,SCALE_LIMIT,\
    ROTATE_LIMIT,ASPECT_LIMIT,U_AUG,TRAIN_SIZE,VAL_SIZE,DR1,DR2,CLASSES,\
    IMG_CLASS,MSK_FLOAT,MSK_THRESHOLD,MRA,MRALEVEL,MRACHANNELS,WAVELET,\
    WAVEMODE,WST,WST_J,WST_L,WST_FIRST_IMG,SCALE_BY_INPUT,SCALE_THRESHOLD = _Params() 
        
    TRAIN_IMG_PATH,TRAIN_MSK_PATH,TRAIN_MSK_CLASS,VAL_IMG_PATH,VAL_MSK_PATH,\
    VAL_MSK_CLASS = _Paths()
    
# In[4s]:
    
    # ### Additional or modified network or fit parameters
    
    NEW_RUN = False
    NEW_MODEL_NUMBER = False
    
    UPSAMPLE = False
    UPSAMPLE_KERNEL = (2,2)
    
    KS1=(3, 3)
    KS2=(3, 3)
    KS3=(3, 3)
    
    WSTCHANNELS = 50
    
    RESNET_DIM_1 = 75
    RESNET_DIM_2 = 85
    
    SCALE_BY_INPUT = False
    SCALE_THRESHOLD = 0.6
    SCALE_TO_SPAN = False
    SPAN = 1.0
    
    ATT = 'mh'
    HEAD_SIZE = 64
    NUM_HEAD = 2 
    VALUE_ATT = True
    
    BLUR_ATT = False
    BLUR_ATT_STD = 0.1
    BLUR_SBI = False
    BLUR_SBI_STD = 0.1
    
    NR1 = 2
    
    PREP = True
    STEM = True
    
    KFOLD = 'Simple' # 'Simple','Strati','Group'
    
    VAL_SIZE = 15
    
    OPTIMIZER = Adam(learning_rate=0.002,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    amsgrad=True)
    
# In[5s]:
    
    model_selection = 'model_' + str(NF) + 'F_' + str(NR1) + 'R1_' + str(NR2) + 'R2'
    
    if NEW_MODEL_NUMBER:
        model_number = str(datetime.datetime.now())[0:10] + '_' + \
        str(datetime.datetime.now())[11:13] + '_' + \
        str(datetime.datetime.now())[14:16]
    else:
        model_number = '2021-02-16_11_28'
        print(f'Model number: {model_number}')    
    
    # ###  LOAD MODELS AND COMPILE
    if ('ensemble_model' in locals()) or ('ensemble_model' in globals()):
        print('Ensemble model already loaded')
    else:
        print('Loading and compiling ensemble model')        
    
        # ### ENSEMBLE MODEL
        K.clear_session()
        
        if SCALE_BY_INPUT:
            loi = 'multiply_2'
        else:
            loi = 'multiply_1'
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
          
            # MODELS
        
            wst_model = WaveletScatteringTransform(input_shape=(HEIGHT, WIDTH, CHANNELS),
                                                    upsample=UPSAMPLE,
                                                    upsample_kernel=UPSAMPLE_KERNEL)
            
            # wst_model.save('models/wst_model')
            
            # Alternatively, load the saved wst_model.
            # wst_model = load_model('models/wst_model')    
        
            resnet_model = ResNet(input_shape_1=(RESNET_DIM_1, RESNET_DIM_2, WSTCHANNELS),
                                    input_shape_2=(RESNET_DIM_1, RESNET_DIM_2, 1),
                                    num_class=NUM_CLASS,
                                    ks1=KS1, ks2=KS2, ks3=KS3, 
                                    dl1=DL1, dl2=DL2, dl3=DL3,
                                    filters=NF,resblock1=NR1,
                                    r_filters=NFL, resblock2=NR2,
                                    dil_mode=DIL_MODE, 
                                    sp_dropout=DR1,re_dropout=DR2,
                                    prep=PREP,
                                    stem=STEM,
                                    mask_float=MSK_FLOAT,
                                    mask_threshold=MSK_THRESHOLD,
                                    att=ATT,
                                    head_size=HEAD_SIZE,
                                    num_heads=NUM_HEAD,
                                    value_att=VALUE_ATT,
                                    scale_by_input=SCALE_BY_INPUT,
                                    scale_threshold=SCALE_THRESHOLD,
                                    scale_to_span=SCALE_TO_SPAN,
                                    span=SPAN,                          
                                    blur_sbi=BLUR_SBI,
                                    blur_sbi_std=BLUR_SBI_STD,                                                                 
                                    return_seq=True)
        
            # recover individual resnet models
            resnet_model_0 = clone_model(resnet_model)
            resnet_model_0.load_weights('models/' + model_selection + '_' + model_number + '_M0' + '_resnet_weights.h5')
        
            
            for layer in resnet_model_0.layers:
                layer.trainable = False                
            resnet_model__0 = Model(inputs=[resnet_model_0.inputs], 
                                    outputs=[resnet_model_0.get_layer(loi).output])  
        
            
            resnet_model_1 = clone_model(resnet_model)
            resnet_model_1.load_weights('models/' + model_selection + '_' + model_number + '_M1' + '_resnet_weights.h5')
            
            for layer in resnet_model_1.layers:
                layer.trainable = False                
            resnet_model__1 = Model(inputs=[resnet_model_1.inputs], 
                                    outputs=[resnet_model_1.get_layer(loi).output]) 
        
            
            resnet_model_2 = clone_model(resnet_model)
            resnet_model_2.load_weights('models/' + model_selection + '_' + model_number + '_M2' + '_resnet_weights.h5')
            
            for layer in resnet_model_2.layers:
                layer.trainable = False                
            resnet_model__2 = Model(inputs=[resnet_model_2.inputs], 
                                    outputs=[resnet_model_2.get_layer(loi).output])
        
            
            resnet_model_3 = clone_model(resnet_model)
            resnet_model_3.load_weights('models/' + model_selection + '_' + model_number + '_M3' + '_resnet_weights.h5')
            
            for layer in resnet_model_3.layers:
                layer.trainable = False                
            resnet_model__3 = Model(inputs=[resnet_model_3.inputs], 
                                    outputs=[resnet_model_3.get_layer(loi).output]) 
        
            
            resnet_model_4 = clone_model(resnet_model)
            resnet_model_4.load_weights('models/' + model_selection + '_' + model_number + '_M4' + '_resnet_weights.h5')
            
            for layer in resnet_model_4.layers:
                layer.trainable = False                
            resnet_model__4 = Model(inputs=[resnet_model_4.inputs], 
                                    outputs=[resnet_model_4.get_layer(loi).output])     
        
            
            resnet_model_5 = clone_model(resnet_model)
            resnet_model_5.load_weights('models/' + model_selection + '_' + model_number + '_M5' + '_resnet_weights.h5')
            
            for layer in resnet_model_5.layers:
                layer.trainable = False                
            resnet_model__5 = Model(inputs=[resnet_model_5.inputs], 
                                    outputs=[resnet_model_5.get_layer(loi).output]) 
            
        
            # GRAPH 1
            
            wst_input_1 = Input(shape=(HEIGHT, WIDTH, CHANNELS))
            wst_input_2 = Input(shape=(HEIGHT, WIDTH, CHANNELS)) 
            
            wst_output_1 = wst_model([wst_input_1,wst_input_2])    
            
            y0 = resnet_model__0(wst_output_1)
            y1 = resnet_model__1(wst_output_1)    
            y2 = resnet_model__2(wst_output_1)
            y3 = resnet_model__3(wst_output_1)
            y4 = resnet_model__4(wst_output_1)
            y5 = resnet_model__5(wst_output_1)    
            
            d3 = Average()([y0,y1,y2,y3,y4,y5])     
            
            d3 = GlobalAveragePooling2D()(d3)
            
            resnet_output = Activation("softmax", name = 'softmax')(d3)     
            
        
            ensemble_model = Model([wst_input_1,wst_input_2], resnet_output,name='ensemble_wst_resnet')
        
            # Save ensemble model in TF2 ModelSave format
            # ensemble_model.save('models/ensemble_model')    
            # ensemble_model = load_model('models/ensemble_model')   
            # Save ensemble model in json format
            # config = ensemble_model.to_json()    
        
            ensemble_model.compile(optimizer=Adam(), 
                          loss=tf.keras.losses.CategoricalCrossentropy(), 
                          metrics=[tf.keras.metrics.CategoricalAccuracy()]) 
        
            # Save ensemble model in TF2 ModelSave format
            # ensemble_model.save('models/ensemble_model')
            # Save ensemble model in h5 format    
            # ensemble_model.save('models/ensemble_model.h5')
     
    
# In[6s]:
    
    # ### GENERATOR for 1 IMAGE at a time
    
    datadir = os.path.join(H5_IMAGE_DIR, '')
    
    dataset = new_patient_h5_dict
                                          
    valid_1_generator = DataGenerator(dataset["new_patient"], datadir, augment=False, shuffle=False, standard=False,                                      batch_size=1, dim=(HEIGHT, WIDTH, MRACHANNELS), mask_dim=(HEIGHT, WIDTH, 1),                                       mlDWT=False, mralevel=MRALEVEL, wave=WAVELET, wavemode=WAVEMODE, verbose=0)
    
# In[7s]:
    
    # ### PREDICT
    
    valid_y_pred = []
    
    for i in range(len(dataset["new_patient"])):
        x_m, y, w = valid_1_generator.__getitem__(i)
        y_pred = ensemble_model(x_m).numpy().tolist()
        valid_y_pred.append(y_pred[0])
        
    # save predictions as dictionary
    new_patient_scores = {"new_patient_scores":valid_y_pred}
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_scores.json'), 'w') as filehandle:
        json.dump(new_patient_scores, filehandle)     
        
    valid_y_pred = np.array(valid_y_pred)
        
# In[8s]:
    
    patient_no = 0
    for idx,patient in enumerate(new_patient_h5_dict["new_patient"]):
        print(f'{patient[:-3]} scores: {valid_y_pred[idx]}')
    
    new_patient_list = np.array(dataset["new_patient"]).tolist()
        
###-------------------------End of Get_scores subroutine----------------------        

    # H5_IMAGE_DIR = os.path.join(entries[0], entries[6])
    # with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle_1:
    #     new_patient_h5_dict = json.load(filehandle_1)  
        
    # with open(os.path.join(H5_IMAGE_DIR, 'new_patient_scores.json'),'r') as filehandle_2:
    #     new_patient_scores = json.load(filehandle_2)    

    dirlist.insert(END,'Scores calculation completed')
    dirlist.insert(END,'')     
    
    for idx,row_name in enumerate(new_patient_h5_dict["new_patient"]):
        pos_score = str(new_patient_scores["new_patient_scores"][idx][0])
        neg_score = str(new_patient_scores["new_patient_scores"][idx][1])        
        row = row_name[:-3] + ' scores: [' + pos_score[:6] + ',' + neg_score[:6] + ']'
        dirlist.insert(END,row)

    dirlist.insert(END,'')                  

button_11_0 = Button(main_window,text='Get scores',font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=10,
       command=get_scores)
button_11_0.grid(row=11,column=0)


# BUTTON_12_0
def get_heat_maps():

    global entries, root_dir, dcm_source_img_path, source_resized_img_path, \
        target_resized_msk_path_binary, target_resized_msk_path_float,\
        target_img_mask_path, h5_img_dir, H5_IMAGE_DIR, grad_cam_dir, valid_image_name,\
        new_patient_h5_dict, new_patient_dcm_name_list, ensemble_model,\
        valid_1_generator, new_patient_list
        
# In[2s]:
    
    # PREDICTION and HEAT MAP
    
    from MODULES_2.Constants import _Params
    from MODULES_2.Utils import _HEAT_MAP_DIFF   
            
    HEIGHT,WIDTH,CHANNELS,IMG_COLOR_MODE,MSK_COLOR_MODE,NUM_CLASS,KS1,KS2,KS3,\
    DL1,DL2,DL3,NF,NFL,NR1,NR2,DIL_MODE,W_MODE,LS,SHIFT_LIMIT,SCALE_LIMIT,\
    ROTATE_LIMIT,ASPECT_LIMIT,U_AUG,TRAIN_SIZE,VAL_SIZE,DR1,DR2,CLASSES,\
    IMG_CLASS,MSK_FLOAT,MSK_THRESHOLD,MRA,MRALEVEL,MRACHANNELS,WAVELET,\
    WAVEMODE,WST,WST_J,WST_L,WST_FIRST_IMG,SCALE_BY_INPUT,SCALE_THRESHOLD = _Params()
    
    # dirlist.delete(0,END)
    # dirlist.insert(END,'')    
    dirlist.insert(END,'Calculating heat maps') 
    
    # ### HEAT MAPS
    
    print('Calculating heat maps')
    
    
    OUT_IMAGE_DIR = os.path.join(grad_cam_dir,'')
    
    
    FIG_SIZE = (16,20)
    
    _HEAT_MAP_DIFF(ensemble_model,
                    generator=valid_1_generator,
                    layer='average',          
                    labels=['Positive score','Negative score'],
                    header='',
                    figsize=FIG_SIZE,          
                    image_dir=H5_IMAGE_DIR,out_image_dir=OUT_IMAGE_DIR,          
                    img_list=new_patient_list,first_img=0,last_img=len(new_patient_list),          
                    img_width=WIDTH,img_height=HEIGHT,display=True)      

    dirlist.insert(END,'Heat maps calculation completed')
    dirlist.insert(END,'')     

button_12_0 = Button(main_window,text='Get maps',font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=10,
       command=get_heat_maps)
button_12_0.grid(row=12,column=0)    
    
# BUTTON_13_0
def quit():
    main_window.destroy()
          
button_13_0 = Button(main_window, text='QUIT', 
        font=('Helvetica',10,'bold'),
        bg='yellow',fg='red',activebackground='orange',activeforeground='red',
        height=3,width=10,command=quit)
button_13_0.grid(row=13,column=0)


# BUTTON_9_4
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img1():
    global panel1
    
    scale = .8
    x = openfn()
    # print(x[-13:-6])
    if x[-3:] == 'dcm':  
        # filename = os.path.join(dcm_source_img_path, name)
        dataset = pydicom.dcmread(x)    
                            
        # Write PNG image    
        img = dataset.pixel_array.astype(float)    
        minval = np.min(img)
        maxval = np.max(img)
        scaled_img = (img - minval)/(maxval-minval) * 255.0
        
        WIDTH = int(np.floor(340*scale))
        HEIGHT = int(np.floor(300*scale))
    
        resized_img = cv2.resize(scaled_img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
        resized_img_8bit = cv2.convertScaleAbs(resized_img, alpha=1.0)
        equalized_img = cv2.equalizeHist(resized_img_8bit)
        img = Image.fromarray(equalized_img)
        
    elif x[-3:] == 'png':        
        img = Image.open(x)
        if x[-24:-17] == 'heatmap':
            WIDTH = int(np.floor(1008*.9))
            HEIGHT = int(np.floor(224*.9))
        elif x[-13:-4] == 'pred_mask':
            WIDTH = int(np.floor(1020*scale))
            HEIGHT = int(np.floor(300*scale))
        else:
            WIDTH = int(np.floor(340*scale))
            HEIGHT = int(np.floor(300*scale))                  
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            
    current_img = ImageTk.PhotoImage(img) 
    panel1 = Label(main_window, image=current_img)
    panel1.configure(image=current_img)
    panel1.image = current_img         
    panel1.grid(row=2,column=3,rowspan=7,columnspan=3)
    
    # Deactivate button after selection to avoid img superposition
    button_9_4['state'] = DISABLED    

button_9_4 = Button(main_window, text='Display CXR/mask', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=30,command=open_img1)
button_9_4.grid(row=9,column=4,padx=20)


# BUTTON_9_5
def delete_img1():
    panel1.grid_forget()
    # panel.destroy()
    button_9_4['state'] = NORMAL    
          
button_9_5 = Button(main_window, text='Delete CXR/mask', 
        font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=30,command=delete_img1)
button_9_5.grid(row=9,column=5,padx=20)

# BUTTON_13_4
def open_img2():
    global panel2

    scale = .8    
    x = openfn()
    # print(x[-13:-6])
    if x[-3:] == 'dcm':  
        dataset = pydicom.dcmread(x)    
                            
        # Write PNG image    
        img = dataset.pixel_array.astype(float)    
        minval = np.min(img)
        maxval = np.max(img)
        scaled_img = (img - minval)/(maxval-minval) * 255.0
        
        WIDTH = int(np.floor(340*scale))
        HEIGHT = int(np.floor(300*scale))
    
        resized_img = cv2.resize(scaled_img, (WIDTH, HEIGHT), cv2.INTER_LINEAR)
        resized_img_8bit = cv2.convertScaleAbs(resized_img, alpha=1.0)
        equalized_img = cv2.equalizeHist(resized_img_8bit)
        img = Image.fromarray(equalized_img)
        
    elif x[-3:] == 'png':        
        img = Image.open(x)
        if x[-24:-17] == 'heatmap':
            WIDTH = int(np.floor(1008*.9))
            HEIGHT = int(np.floor(224*.9))
        elif x[-13:-4] == 'pred_mask':
            WIDTH = int(np.floor(1020*scale))
            HEIGHT = int(np.floor(300*scale))
        else:
            WIDTH = int(np.floor(340*scale))
            HEIGHT = int(np.floor(300*scale))                          
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            
    current_img = ImageTk.PhotoImage(img) 
    panel2 = Label(main_window, image=current_img)
    panel2.configure(image=current_img)
    panel2.image = current_img         
    panel2.grid(row=10,column=3,rowspan=3,columnspan=3)

    # Deactivate button after selection to avoid img superposition
    button_13_4['state'] = DISABLED    

button_13_4 = Button(main_window, text='Display heat map', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=30,command=open_img2)
button_13_4.grid(row=13,column=4,padx=20)

# BUTTON_13_5
def delete_img2():
    panel2.grid_forget()
    # panel2.destroy()
    button_13_4['state'] = NORMAL     
          
button_13_5 = Button(main_window, text='Delete heat map', 
        font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=30,command=delete_img2)
button_13_5.grid(row=13,column=5,padx=20)


# BUTTON TO CLEAR ALL DIRECTORIES (currently unused)

def clear_all():
    # Remove existing target directories and all their content if already present
    pwd = os.getcwd()
    
    if root_dir == pwd:
        for root, dirs, files in os.walk(source_resized_img_path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))    
        for root, dirs, files in os.walk(target_resized_msk_path_binary):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        for root, dirs, files in os.walk(target_resized_msk_path_float):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d)) 
        for root, dirs, files in os.walk(target_img_mask_path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        for root, dirs, files in os.walk(h5_img_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        for root, dirs, files in os.walk(grad_cam_dir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))            
    
    # Create directory that will store the DCM derived png CXR
    if not os.path.exists(source_resized_img_path):
        os.makedirs(source_resized_img_path)            
                
    # Create directories that will store the masks on which to train the classification network
    if not os.path.exists(target_resized_msk_path_binary):
        os.makedirs(target_resized_msk_path_binary)
        
    if not os.path.exists(target_resized_msk_path_float):
        os.makedirs(target_resized_msk_path_float) 
        
    if not os.path.exists(target_img_mask_path):
        os.makedirs(target_img_mask_path)
    
    # Create directory that will store the H5 files
    if not os.path.exists(h5_img_dir):
        os.makedirs(h5_img_dir)
    
    # Create directory that will store the heat maps
    if not os.path.exists(grad_cam_dir):
        os.makedirs(grad_cam_dir)
    

main_window.mainloop()
