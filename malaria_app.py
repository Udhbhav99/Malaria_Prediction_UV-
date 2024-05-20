import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from keras_preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import os
def load_keras_model():
    model = load_model('model_vgg19.h5')  # Load your Keras model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

icon= Image.open('mosquito 1.jpg')
st.set_page_config(page_title='Malaria prediction', page_icon=icon,layout='wide')
st.markdown(f""" <style>.stApp {{
                    background:url("https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20211027100544/ri/673/picture/2021/10/shutterstock_1483138139-1.jpg");
                    background-size: 35%}}
                    
                 </style>""", unsafe_allow_html=True)
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
c1, c2, c3 = st.columns([1, 2,1])
with c2:
    uploaded_img = st.file_uploader("upload", label_visibility="collapsed", type=["png", "jpeg", "jpg"])

model=load_keras_model()
if uploaded_img is not None:
    # Display the uploaded image

    image = Image.open(uploaded_img)
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=False)

    # Preprocess the image
      # Resize the image to match the input size of the model
    image = np.asarray(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    img_data = preprocess_input(image)

    pred=np.argmax(model.predict(img_data),axis=1)

    if pred==1:
        st.header("You are not infected")
        st.subheader("the accuracy of the prediction is :red[86.32%]")
    else:
        st.header("You are likely infected with malaria, please consult the doctor")
        st.subheader("the accuracy of the prediction is :red[86.32%]")
