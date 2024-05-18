import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import streamlit as st

def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image(image_path):
    img = load_image(image_path)
    img = preprocess_input(img)
    return img

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    input_tensor = tf.concat([init_image, content_features, gram_style_features], axis=0)
    features = model(input_tensor)
    style_features = features[:len(gram_style_features)]
    content_features = features[len(gram_style_features):]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(len(style_features))
    for target_style, comb_style in zip(gram_style_features, style_features):
        style_score += weight_per_style_layer * tf.reduce_mean(tf.square(comb_style - target_style))

    weight_per_content_layer = 1.0 / float(len(content_features))
    for target_content, comb_content in zip(content_features, content_features):
        content_score += weight_per_content_layer * tf.reduce_mean(tf.square(comb_content - target_content))

    style_score *= loss_weights[0]
    content_score *= loss_weights[1]

    loss = style_score + content_score
    return loss, style_score, content_score

@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = preprocess_image(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

    return best_img

def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = Model([vgg.input], outputs)
    return model

def get_feature_representations(model, content_path, style_path):
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    style_features = [style_layer[0] for style_layer in style_outputs[:5]]
    content_features = [content_layer[0] for content_layer in content_outputs[5:]]

    return style_features, content_features

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Streamlit app
st.title("Neural Style Transfer")

content_file = st.file_uploader("Choose a Content Image", type=["jpg", "png"])
style_file = st.file_uploader("Choose a Style Image", type=["jpg", "png"])

if content_file and style_file:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)

    st.image(content_image, caption="Content Image", use_column_width=True)
    st.image(style_image, caption="Style Image", use_column_width=True)
    
    if st.button("Run Style Transfer"):
        content_path = content_file
        style_path = style_file
        
        output_image = run_style_transfer(content_path, style_path)
        output_image = deprocess_image(output_image)
        st.image(output_image, caption="Output Image", use_column_width=True)
