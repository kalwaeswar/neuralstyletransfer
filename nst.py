import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Function to load and preprocess images
def load_and_preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")    
    img = img.save("temp.jpg")
    img = cv2.imread("./temp.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    os.remove("./temp.jpg")
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    return img

# Function to resize images
def resize_image(image, target_shape=(256, 256)):
    return cv2.resize(image, target_shape)

# Function to preprocess images for the model
def preprocess_image(img):
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert dtype
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Function to display images
def display_image(image):
    st.image(image, use_column_width=True)

# Function to perform neural style transfer
def neural_style_transfer(content_image, style_image, content_weight=1e3, style_weight=1e-2, total_variation_weight=30):
    # Load pre-trained VGG19 model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # Extract content and style features
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    content_outputs = [vgg.get_layer(layer).output for layer in content_layers]
    style_outputs = [vgg.get_layer(layer).output for layer in style_layers]

    model_outputs = content_outputs + style_outputs

    # Build model
    model = tf.keras.models.Model(vgg.input, model_outputs)

    # Calculate content loss
    def content_loss(content, target):
        return tf.reduce_mean(tf.square(content - target))

    # Calculate style loss
    def style_loss(style, target):
        style = tf.reshape(style, (style.shape[0], -1))
        target = tf.reshape(target, (target.shape[0], -1))
        style_gram = tf.matmul(style, style, transpose_a=True)
        target_gram = tf.matmul(target, target, transpose_a=True)
        return tf.reduce_mean(tf.square(style_gram - target_gram))

    # Calculate total variation loss
    def total_variation_loss(image):
        x_deltas, y_deltas = tf.image.image_gradients(image)
        return tf.reduce_mean(tf.square(x_deltas)) + tf.reduce_mean(tf.square(y_deltas))

    # Run optimization
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = model(image)
            content_output_features = outputs[:len(content_layers)]
            style_output_features = outputs[len(content_layers):]

            content_score = 0
            style_score = 0

            # Calculate content loss
            weight_per_content_layer = 1.0 / float(len(content_layers))
            for target_content, comb_content in zip(content_outputs, content_output_features):
                content_score += weight_per_content_layer * content_loss(comb_content[0], target_content)

            # Calculate style loss
            weight_per_style_layer = 1.0 / float(len(style_layers))
            for target_style, comb_style in zip(style_outputs, style_output_features):
                style_score += weight_per_style_layer * style_loss(comb_style[0], target_style)

            # Calculate total loss
            loss = content_weight * content_score + style_weight * style_score + total_variation_weight * total_variation_loss(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

    # Initialize generated image
    generated_image = tf.Variable(content_image)

    # Define optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Training loop
    for i in range(100):
        train_step(generated_image)

    return generated_image

# Streamlit app
def main():
    st.title("Neural Style Transfer")

    content_image = st.file_uploader("Upload content image", type=['png', 'jpeg', 'jpg'])
    style_image = st.file_uploader("Upload style image", type=['png', 'jpeg', 'jpg'])

    if content_image and style_image:
        content_array = load_and_preprocess_image(content_image)
        style_array = load_and_preprocess_image(style_image)

        st.write("Content Image:")
        display_image(content_array)

        st.write("Style Image:")
        display_image(style_array)

        st.write("Processing...")

        generated_image = neural_style_transfer(preprocess_image(content_array), preprocess_image(style_array))

        st.write("Generated Image:")
        display_image(generated_image.numpy()[0])

if __name__ == '__main__':
    main()
