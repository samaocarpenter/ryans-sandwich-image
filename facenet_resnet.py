#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### Take image and turn it into some descriptors for that image
###def describe_image(image):
###pass the image --- possibly reshape into into (R, C, 3)??
###pass image turn model (instance of facenet class)
###get boxes, probabilities, and landmarks
##pass boxes and image through resnet and get descriptors
##return descriptors, probabilites, and landmarks


# In[49]:


from facenet_models import FacenetModel
from pathlib import Path
from camera import take_picture
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image




def describe_image(image):
    """Takes image, reshapes it, and returns descriptors, 
    probabilites, and landmarks
    
    
    Params
    --------
    image shape-(N, H, W, 3)
    
    
    
    Returns
    --------
    descriptors - np.ndarray, shape=(N, 512)
                  The descriptor vectors, where N is the number of faces.
                  
    probabilities - shape-(N,)  looks like: ([%%], dtype=float)
                    array of probabilities corresponding to each detected face
                    
    landmarks - shape-(N, 5, 2) 
                arrays of facial landmarks corresponding to each detected face."""
    #img_reshaped = image.reshape()
    model = FacenetModel()
    boxes, probabilities, landmarks = model.detect(image) #model detect returns boxes, prob%%, and landmark
    descriptors = model.compute_descriptors(image, boxes)
    return descriptors, probabilities, landmarks


# In[47]:


#descriptor, prob, landmarks = describe_image(img)
#landmarks.shape


# In[ ]:




