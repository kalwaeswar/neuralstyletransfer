import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import os
import tensorflow_hub as hub
import tensorflow as tf
import cv2
>>>>>>> 02baac4042986e339edb919dc4dc10266e72e98f
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.title("Neural Style Transfer App")

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    return hub_model

model = load_model()

# Function to preprocess images
def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))  # Resize image to match model input size
    return image[tf.newaxis, :]

# Function to perform style transfer
def style_transfer(content_image, style_image):
    content_image = preprocess_image(content_image)
    style_image = preprocess_image(style_image)
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image.numpy()

# Sidebar - Upload images
st.sidebar.header('Upload Images')
content_file = st.sidebar.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
style_file = st.sidebar.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

if content_file is not None and style_file is not None:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)
    
    st.sidebar.image(content_image, caption='Content Image', use_column_width=True)
    st.sidebar.image(style_image, caption='Style Image', use_column_width=True)
    
    # Perform style transfer
    with st.spinner('Performing style transfer...'):
        stylized_image = style_transfer(content_image, style_image)
    
    st.subheader('Stylized Image')
    st.image(stylized_image, caption='Stylized Image', use_column_width=True)
else:
    st.info("Please upload content and style images on the sidebar.")
