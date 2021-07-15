#!/usr/bin/env python
# coding: utf-8

# In[31]:


import skimage.io as io
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
from camera import take_picture
import matplotlib.pyplot as plt
import skimage.io as io

def get_image() :

    type = input("Camera or Image?: ")

    if type.lower() == "camera": 
        image = take_picture()  # returns shape-(H, W, C) array
    
    elif type.lower() == "image":
        path = str(input("Input a path: "))
        pathname = r'{}'.format(path)
        image = io.imread(pathname)
        if image.shape[-1] == 4:
            # Image is RGBA, where A is alpha -> transparency
            # Must make image RGB.
            image = image[..., :-1]
        
    fig,ax = plt.subplots()
    ax.imshow(image)
    check_or_load = input("Are you loading or checking your image?")

    return image, check_or_load





