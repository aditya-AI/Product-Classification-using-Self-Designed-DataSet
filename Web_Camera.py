#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import sys
import threading
from keras import models


# In[ ]:


label = ''
frame = None

num_to_label = {0:'ThumbsUp Can',1:'Mirinda Bottle',2:'Tide Packet',3:'Coke Can'}


def preprocess(img):
    img = (img - np.mean(img)) / np.std(img)
    return img



class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        print("Loading Model")
        self.model = models.load_model('resnet_all.h5')
        print("Model Loaded Successfully")

        while (~(frame is None)):
            label = self.predict(frame)

    def predict(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = preprocess(image)
        image = image.reshape(-1,224,224,3)
        pred = np.argmax(self.model.predict(image),axis=1)
        label = [num_to_label[item] for item in pred]
        label = "".join(label)
        return label

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))

    cv2.putText(original, "PRODUCT LABEL : {}".format(label), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
    cv2.imshow("Product Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()


# In[ ]:




