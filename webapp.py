import requests
import os
import matplotlib.pyplot as plt
import streamlit as st
import json
import numpy as np
import keras as K

URL = 'http://127.0.0.1:5000/server'

st.title('Neural Network Visualizer')
st.sidebar.markdown('# Input Image')


model = K.models.load_model('NNmodel.h5')
feature_model = K.models.Model(model.inputs, [layer.output for layer in model.layers])

_, (x_test, _) = K.datasets.mnist.load_data()
_, w, h = x_test.shape
x_test = x_test / 255.

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, w*h))
    return feature_model.predict(image_arr), image

if st.button('Get random prediction'):
    preds, image = get_prediction()
    final_preds = [p.tolist() for p in preds]
    #image = image.tolist()
    
    preds = final_preds
    image = np.reshape(image, (28, 28))
    
    st.sidebar.image(image, width=150)
    
    for layer, p in enumerate(preds):
        print('p.shape:', np.array(p).shape)
        numbers = np.squeeze(np.array(p))
        print('numbers.shape:', numbers.shape)
        print('layer:', layer)
        plt.figure(figsize=(32, 4))
        
        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16
            
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow((number * np.ones((8, 8, 3))).astype('float32'), cmap='binary')
            plt.xticks([])
            plt.yticks([])
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        
        st.text('Layer {}'.format(layer+1), )
        st.pyplot()