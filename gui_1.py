#!source /Users/dgatti/venv_jupyter/activate.csh
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:11:59 2021

@author: dgatti
"""

# In[]:
import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter import *
# from tkinter.ttk import *
from tkinter import filedialog
from PIL import ImageTk, Image
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import cv2
from threading import Thread
import json

# In[2]:
main_window = Tk()
main_window.title("CXR-Net")
main_window.geometry('1260x680') # 1260x680
main_window.resizable(width=True, height=True)
# main_window.configure(bg='#E5E5E5')


# Labels
title_0_0 = Label(main_window,text='Directories (if none entered defaults are used)',
               font=('Helvetica',12,'bold'),fg='blue')
title_0_0.grid(row=0,column=0,columnspan=2)

title_0_4 = Label(main_window,text='Images (if none entered all images are used)',
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

global entries
global selected_entries
selected_entries = ['','','','','','','','','']


# BUTTON_1_0
def open_dir1():
    selected_entries[0] = selectdir().split('/')[-1]
    print(selected_entries)
    
    dirlist1 = Listbox(main_window,height=15,width=25,borderwidth=0.5,)
    dirlist.grid(row=0,column=1)
    
    
button_1_0 = Button(main_window, text='Root', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir1)
button_1_0.grid(row=1,column=0)

# BUTTON_2_0
def open_dir2():
    selected_entries[1] = selectdir().split('/')[-1]
    
button_2_0 = Button(main_window, text='DCM images', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir2)
button_2_0.grid(row=2,column=0)

# BUTTON_3_0
def open_dir3():
    selected_entries[2] = selectdir().split('/')[-1]
    
button_3_0 = Button(main_window, text='PNG images', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir3)
button_3_0.grid(row=3,column=0)

# BUTTON_4_0
def open_dir4():
    selected_entries[3] = selectdir().split('/')[-1]
    
button_4_0 = Button(main_window, text='Bin. masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir4)
button_4_0.grid(row=4,column=0)

# BUTTON_5_0
def open_dir5():
    selected_entries[4] = selectdir().split('/')[-1]
    
button_5_0 = Button(main_window, text='Flp. masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir5)
button_5_0.grid(row=5,column=0)


# BUTTON_6_0
def open_dir6():
    selected_entries[5] = selectdir().split('/')[-1]
    
button_6_0 = Button(main_window, text='Imgs/masks', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir6)
button_6_0.grid(row=6,column=0)

# BUTTON_7_0
def open_dir7():
    selected_entries[6] = selectdir().split('/')[-1]
    
button_7_0 = Button(main_window, text='H5', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir7)
button_7_0.grid(row=7,column=0)

# BUTTON_8_0
def open_dir8():
    selected_entries[7] = selectdir().split('/')[-1]
    
button_8_0 = Button(main_window, text='Heat maps', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_dir8)
button_8_0.grid(row=8,column=0)


# label_2_0 = Label(main_window,text='DCM images',
#                 font=('Helvetica',10,'bold')).grid(row=2,column=0,sticky=W)
# label_3_0 = Label(main_window,text='PNG images',
#                font=('Helvetica',10,'bold')).grid(row=3,column=0,sticky=W)
# label_4_0 = Label(main_window,text='Bin. masks',
#                font=('Helvetica',10,'bold')).grid(row=4,column=0,sticky=W)
# label_5_0 = Label(main_window,text='Flp. masks',
#                font=('Helvetica',10,'bold')).grid(row=5,column=0,sticky=W)
# label_6_0 = Label(main_window,text='Imgs/masks',
#                font=('Helvetica',10,'bold')).grid(row=6,column=0,sticky=W)
# label_7_0 = Label(main_window,text='H5',
#                font=('Helvetica',10,'bold')).grid(row=7,column=0,sticky=W)
# label_8_0 = Label(main_window,text='Heat maps ',
#                font=('Helvetica',10,'bold')).grid(row=8,column=0,sticky=W)

# BUTTON_1_3   
def open_file9():
    selected_entries[8] = openfns()
    print(selected_entries[8])
    selected_imgs = []
    for selected_name in selected_entries[8]:
        selected_imgs.append(selected_name.split('/')[-1])
    print(selected_imgs)
    
button_1_3 = Button(main_window, text='DCM names', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='red',activebackground='orange',activeforeground='blue',
       height=2,width=12,command=open_file9)
button_1_3.grid(row=1,column=3)

# label_1_3 = Label(main_window,text='DCM names',
#                font=('Helvetica',10,'bold')).grid(row=1,column=3,sticky=W)

# Text input for directories
text_1_1 = StringVar()
entry_1_1 = Entry(main_window,textvariable=text_1_1,width=25,borderwidth=2)
entry_1_1.grid(row=1,column=1,pady=5)

text_2_1 = StringVar()
entry_2_1= Entry(main_window,textvariable=text_2_1,width=25,borderwidth=2)
entry_2_1.grid(row=2,column=1,pady=5)

text_3_1 = StringVar()
entry_3_1 = Entry(main_window,textvariable=text_3_1,width=25,borderwidth=2)
entry_3_1.grid(row=3,column=1,pady=5)

text_4_1 = StringVar()
entry_4_1 = Entry(main_window,textvariable=text_4_1,width=25,borderwidth=2)
entry_4_1.grid(row=4,column=1,pady=5)

text_5_1 = StringVar()
entry_5_1= Entry(main_window,textvariable=text_5_1,width=25,borderwidth=2)
entry_5_1.grid(row=5,column=1,pady=5)

text_6_1 = StringVar()
entry_6_1 = Entry(main_window,textvariable=text_6_1,width=25,borderwidth=2)
entry_6_1.grid(row=6,column=1,pady=5)

text_7_1 = StringVar()
entry_7_1 = Entry(main_window,textvariable=text_7_1,width=25,borderwidth=2)
entry_7_1.grid(row=7,column=1,pady=5)

text_8_1 = StringVar()
entry_8_1 = Entry(main_window,textvariable=text_8_1,width=25,borderwidth=2)
entry_8_1.grid(row=8,column=1,pady=5)

# Text input for images
text_1_4 = StringVar()
entry_1_4 = Entry(main_window,textvariable=text_1_4,width=90,borderwidth=2)
entry_1_4.grid(row=1,column=4,columnspan=2)

# Buttons
 
# BUTTON_9_0
def on_click():        
    entries = [entry_1_1.get(),entry_2_1.get(),entry_3_1.get(),\
                  entry_4_1.get(),entry_5_1.get(),entry_6_1.get(),\
                  entry_7_1.get(),entry_8_1.get(),entry_1_4.get()]    

    if selected_entries[0]!='': entries[0]=selected_entries[0]
    if selected_entries[1]!='': entries[1]=selected_entries[1]
    if selected_entries[2]!='': entries[2]=selected_entries[2]
    if selected_entries[3]!='': entries[3]=selected_entries[3]
    if selected_entries[4]!='': entries[4]=selected_entries[4]
    if selected_entries[5]!='': entries[5]=selected_entries[5]
    if selected_entries[6]!='': entries[6]=selected_entries[6]
    if selected_entries[7]!='': entries[7]=selected_entries[7]
    if selected_entries[8]!='': entries[8]=selected_imgs    

    if entries[0]=='': entries[0]='new_patient_cxr'
    if entries[1]=='': entries[1]='image_dcm'
    if entries[2]=='': entries[2]='image_resized_equalized_from_dcm'
    if entries[3]=='': entries[3]='mask_binary'
    if entries[4]=='': entries[4]='mask_float'
    if entries[5]=='': entries[5]='image_mask'
    if entries[6]=='': entries[6]='H5'
    if entries[7]=='': entries[7]='grad_cam'
    
    print(entries[8])
        
    selected_imgs = entries[8].split()
    
    print(selected_imgs)
    

    # ### Generate json dictionary for the new patient names
    # DCM directory containing COVID patients lung CXR's
    dcm_source_img_path = os.path.join(entries[0], entries[1]) # 'new_patient_cxr/image_dcm/'    

    source_img_names = [f for f in listdir(dcm_source_img_path) if isfile(join(dcm_source_img_path, f))]    
    print(source_img_names)
    
    new_patient_dcm_name_list = []
    
    for valid_image_name in source_img_names:
        if valid_image_name == '.DS_Store': 
            continue
        if (entries[8]!='') and (valid_image_name not in selected_imgs):
            continue    

        valid_image_name = valid_image_name[:-3] + 'h5'
        new_patient_dcm_name_list.append(valid_image_name) 
    
    new_patient_h5_dict = {"new_patient":new_patient_dcm_name_list}
    
    # data set
    H5_IMAGE_DIR = os.path.join(entries[0], entries[6])
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'), 'w') as filehandle:
        json.dump(new_patient_h5_dict, filehandle)     
    
    # print(f'H5 dataset: {new_patient_h5_dict["new_patient"]}')

    
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

dirlist = Listbox(main_window,height=15,width=25,borderwidth=0.5)
dirlist.grid(row=9,column=1,rowspan=3,columnspan=2,pady=20)
vscrollbar = Scrollbar(main_window,orient=VERTICAL)
vscrollbar.grid(row=9,column=3,rowspan=3,padx=20)
dirlist.configure(yscrollcommand=vscrollbar.set)
vscrollbar.configure(command=dirlist.yview)
hscrollbar = Scrollbar(main_window,orient=HORIZONTAL)
hscrollbar.grid(row=12,column=1,columnspan=2)
dirlist.configure(xscrollcommand=hscrollbar.set)
hscrollbar.configure(command=dirlist.xview)
        
button_9_0 = Button(main_window,text='Update',font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=10,
        command=on_click)
button_9_0.grid(row=9,column=0,pady=10)


# BUTTON_10_0
def get_masks_threaded():
    Thread(target=get_masks).start()


def get_masks():
    
    entries = [entry_1_1.get(),entry_2_1.get(),entry_3_1.get(),\
                  entry_4_1.get(),entry_5_1.get(),entry_6_1.get(),\
                  entry_7_1.get(),entry_8_1.get(),entry_1_4.get()]    

    if entries[0]=='': entries[0]='new_patient_cxr'
    if entries[1]=='': entries[1]='image_dcm'
    if entries[2]=='': entries[2]='image_resized_equalized_from_dcm'
    if entries[3]=='': entries[3]='mask_binary'
    if entries[4]=='': entries[4]='mask_float'
    if entries[5]=='': entries[5]='image_mask'
    if entries[6]=='': entries[6]='H5'
    if entries[7]=='': entries[7]='grad_cam'
    
    print(entries)
    
    # Read in 'new patient list'
    H5_IMAGE_DIR = os.path.join(entries[0], entries[6])
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle:
        new_patient_h5_dict = json.load(filehandle)     

    dirlist.delete(0,END)
    # for row in entries[-1]:
    for row in new_patient_h5_dict["new_patient"]:
        dirlist.insert(END,row[:-3])    

    if entries[-1] == '':       
        command_line = 'python Get_masks.py --dirs ' + ' ' + entries[0] \
                                   + ' ' + entries[1] \
                                   + ' ' + entries[2] + ' ' + entries[3] \
                                   + ' ' + entries[4] + ' ' + entries[5] \
                                   + ' ' + entries[6] + ' ' + entries[7]                                  
    else:
        command_line = 'python Get_masks.py --dirs ' + ' ' + entries[0] \
                                   + ' ' + entries[1] \
                                   + ' ' + entries[2] + ' ' + entries[3] \
                                   + ' ' + entries[4] + ' ' + entries[5] \
                                   + ' ' + entries[6] + ' ' + entries[7] \
                                   + ' ' + '--imgs' + ' ' + entries[8]
                              
        # print(f'Command syntax: {command_line}')
        
    os.system(command_line)


button_10_0 = Button(main_window,text='Get masks',font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=10,
       command=get_masks_threaded)
button_10_0.grid(row=10,column=0,pady=10)


# BUTTON_11_0
def get_scores_threaded():
    Thread(target=get_scores).start()


def get_scores():
    
    entries = [entry_1_1.get(),entry_2_1.get(),entry_3_1.get(),\
                  entry_4_1.get(),entry_5_1.get(),entry_6_1.get(),\
                  entry_7_1.get(),entry_8_1.get(),entry_1_4.get()]    

    if entries[0]=='': entries[0]='new_patient_cxr'
    if entries[1]=='': entries[1]='image_dcm'
    if entries[2]=='': entries[2]='image_resized_equalized_from_dcm'
    if entries[3]=='': entries[3]='mask_binary'
    if entries[4]=='': entries[4]='mask_float'
    if entries[5]=='': entries[5]='image_mask'
    if entries[6]=='': entries[6]='H5'
    if entries[7]=='': entries[7]='grad_cam'
    
    # print(entries)
          
    command_line = 'python Get_scores.py --dirs ' + ' ' + entries[0] \
                               + ' ' + entries[1] \
                               + ' ' + entries[2] + ' ' + entries[3] \
                               + ' ' + entries[4] + ' ' + entries[5] \
                               + ' ' + entries[6] + ' ' + entries[7]                                  
                             
    print(f'Command syntax: {command_line}')
        
    os.system(command_line)

        
    H5_IMAGE_DIR = os.path.join(entries[0], entries[6])
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_dataset.json'),'r') as filehandle_1:
        new_patient_h5_dict = json.load(filehandle_1)  
        
    with open(os.path.join(H5_IMAGE_DIR, 'new_patient_scores.json'),'r') as filehandle_2:
        new_patient_scores = json.load(filehandle_2)    

    dirlist.delete(0,END)
    
    for idx,row_name in enumerate(new_patient_h5_dict["new_patient"]):
        # row_name = row[:-3]
        pos_score = str(new_patient_scores["new_patient_scores"][idx][0])
        neg_score = str(new_patient_scores["new_patient_scores"][idx][1])        
        row = row_name[:-3] + ' scores: [' + pos_score[:6] + ',' + neg_score[:6] + ']'
        dirlist.insert(END,row) 
        
    # for row in entries:
    #     dirlist.insert(END,row)            

button_11_0 = Button(main_window,text='Get scores',font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=10,
       command=get_scores_threaded)
button_11_0.grid(row=11,column=0,pady=10)


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
        # print(type(img))
        if x[-24:-17] == 'heatmap':
            WIDTH = int(np.floor(1008*.9))
            HEIGHT = int(np.floor(224*.9))
            # print(WIDTH,HEIGHT)
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

button_9_4 = Button(main_window, text='Display CXR/mask', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=30,command=open_img1)
button_9_4.grid(row=9,column=4,padx=20)

# BUTTON_9_5
def delete_img1():
    panel1.grid_forget()
    # panel.destroy()
          
button_9_5 = Button(main_window, text='Delete CXR/mask', 
        font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=30,command=delete_img1)
button_9_5.grid(row=9,column=5,padx=20)

# BUTTON_12_4
# def openfn():
#     filename = filedialog.askopenfilename(title='open')
#     return filename
def open_img2():
    global panel2

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
        # print(type(img))
        if x[-24:-17] == 'heatmap':
            WIDTH = int(np.floor(1008*.9))
            HEIGHT = int(np.floor(224*.9))
            # print(WIDTH,HEIGHT)
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
    panel2.grid(row=10,column=3,rowspan=2,columnspan=3)

button_12_4 = Button(main_window, text='Display heat map', 
       font=('Helvetica',10,'bold'),
       bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
       highlightcolor='yellow',height=3,width=30,command=open_img2)
button_12_4.grid(row=12,column=4,padx=20)

# BUTTON_12_5
def delete_img2():
    panel2.grid_forget()
    # panel2.destroy()
          
button_12_5 = Button(main_window, text='Delete heat map', 
        font=('Helvetica',10,'bold'),
        bg='yellow',fg='green',activebackground='orange',activeforeground='blue',
        highlightcolor='yellow',height=3,width=30,command=delete_img2)
button_12_5.grid(row=12,column=5,padx=20)

main_window.mainloop()